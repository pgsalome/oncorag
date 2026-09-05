import argparse
import requests
import os
import yaml
import json
import time
from dotenv import load_dotenv
import re
import difflib
import math
import sys

from typing import List, Dict, Any
from enum import Enum
from pathlib import Path
from urllib.parse import quote

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oncoraggraph.config.feature_schema import load_feature_specs, generate_feature_configs

wn = None


def _load_wordnet():
    global wn
    try:
        from nltk.corpus import wordnet

        wordnet.ensure_loaded()
    except (ImportError, LookupError, AttributeError):
        raise RuntimeError(
            "Ontology enrichment requires nltk and its WordNet corpus. Install nltk and run "
            "'python -m nltk.downloader wordnet', or choose --mode manual."
        ) from None
    wn = wordnet

# Load environment variables from .env file
load_dotenv()

RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"

# =========================================================
# 📋 FEATURE CATEGORIES
# =========================================================
class FeatureCategory(Enum):
    LAB_TEST = "laboratory_test_biomarker"
    DEMOGRAPHIC = "demographic_social"
    DEMOGRAPHIC_CLINICAL = "demographic_clinical"
    CLINICAL_MEASUREMENT = "clinical_measurement"
    TREATMENT = "treatment_procedure"
    DISEASE = "disease_diagnosis"
    DERIVED = "derived_calculated"
    UNKNOWN = "unknown"

# Target language handling for bilingual config generation
TARGET_LANGUAGE = "english"
OLLAMA_SETTINGS = {}


def set_target_language(language: str):
    """Set global target language for all LLM prompts."""
    global TARGET_LANGUAGE
    TARGET_LANGUAGE = (language or "english").strip().lower()


def language_hint(extra_instruction: str = "") -> str:
    """Return a reusable instruction enforcing the configured language."""
    base = (
        f"Please write all generated text (descriptions, synonyms, keywords, queries, examples) in {TARGET_LANGUAGE}."
    )
    if extra_instruction:
        base = f"{base} {extra_instruction.strip()}"
    return base

EXPECTED_VALUE_SYNONYMS = {
    "yes": {"german": ["ja", "positiv"], "english": ["yes"]},
    "no": {"german": ["nein", "negativ"], "english": ["no"]},
    "left": {"german": ["links", "linke seite", "linksseitig"], "english": ["left"]},
    "right": {"german": ["rechts", "rechte seite", "rechtsseitig"], "english": ["right"]},
    "central": {"german": ["zentral", "mittig", "zentral gelegen"], "english": ["central"]},
    "bilateral": {"german": ["beidseitig", "bilateral"], "english": ["bilateral", "both sides"]},
    "wt": {
        "german": ["wildtyp", "wildtyp (wt)", "wild-type"],
        "english": ["wildtype", "wild-type", "wild type"],
    },
    "mut": {
        "german": ["mutiert", "mutation", "mutierter status"],
        "english": ["mutated", "mutation", "mutant"],
    },
    "unknown": {
        "german": ["unbekannt", "nicht bestimmt", "keine angabe"],
        "english": ["unknown", "not determined", "not reported"],
    },
    "positive": {
        "german": ["positiv", "nachweisbar"],
        "english": ["positive", "detected"],
    },
    "negative": {
        "german": ["negativ", "nicht nachweisbar"],
        "english": ["negative", "not detected"],
    },
}

DATE_KEYWORDS = {
    "birth": ["geburtsdatum", "geboren", "birth date", "dob", "geb."],
    "death": ["sterbedatum", "verstorben", "todestag", "date of death"],
    "diagnosis": ["erstdiagnose", "diagnosedatum", "initial diagnosis", "diagnose am", "erstmanifestation"],
    "progression": ["progression", "progressionsdatum", "fortschritt", "rezidivdatum"],
    "resection_initial": ["erste resektion", "initiale op", "operationsdatum", "resektion am", "erste operation"],
    "resection_repeat": ["re-resektion", "wiederresektion", "revision", "zweite op", "reop", "erneute operation"],
    "carbon_ion_rt": ["carbon-ion", "kohlenstoffionen", "bestrahlungsbeginn", "radiotherapie start", "rt start", "beginn der kohlenstoffionenbestrahlung"],
    "radiation_necrosis": ["strahlennekrose", "radiation necrosis", "nekrose", "diagnose", "datum"],
}

LATERALITY_TERMS = ["links", "rechts", "bilateral", "beidseitig", "zentral", "linksseitig", "rechtsseitig", "beiden hemisphären", "mittellinie"]

TOPOGRAPHY_TERMS = [
    "c71.0", "c71.1", "c71.2", "c71.3", "c71.4", "c71.5", "c71.6", "c71.7", "c71.8", "c71.9",
    "frontallappen", "temporallappen", "parietallappen", "okzipitallappen", "insula", "thalamus",
    "basalganglien", "zentral", "kleinhirn", "cerebellum", "hirnstamm", "überlappend", "multifokal"
]

RT_DOSE_TERMS = ["gy", "gray", "cgy", "dosis", "gesamtdosis", "fraktionen", "rt-dose", "bestrahlungsdosis", "strahlendosis"]

MED_SYNONYM_SEEDS = {
    "bevac": ["bevacizumab", "avastin", "bevac", "anti-vegf"],
    "avast": ["bevacizumab", "avastin", "bevac", "anti-vegf"],
    "dexam": ["dexamethason", "dexamethasone", "decadron", "cortison", "corticosteroid"],
    "cortiso": ["dexamethason", "dexamethasone", "decadron", "cortison", "corticosteroid"],
    "ccnu": ["ccnu", "lomustin", "lomustine", "gleostine"],
    "lomust": ["ccnu", "lomustin", "lomustine", "gleostine"],
    "pcv": ["pcv", "procarbazin", "ccnu", "vincristin"],
    "vp16": ["vp-16", "etoposid", "ccnu/vp16", "etoposide"],
    "temoz": ["temozolomid", "temozolomide", "temodal"],
    "fotem": ["fotemustin", "fotemustine"],
    "procarb": ["procarbazin", "procarbazine"],
    "nivol": ["nivolumab"],
    "pembro": ["pembrolizumab"],
    "methadon": ["methadon", "methadone"],
}

# =========================================================
# 🧩 LLM UTILITIES
# =========================================================
def run_ollama(prompt: str, model: str | None = None) -> str:
    """Run Ollama locally to generate text"""
    host = OLLAMA_SETTINGS.get("host") or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model or OLLAMA_SETTINGS.get("model") or os.getenv("OLLAMA_MODEL", "llama3.1:70b"),
        "prompt": prompt,
        "stream": False,
    }
    options = {key: OLLAMA_SETTINGS[key] for key in ("temperature", "num_ctx", "seed") if OLLAMA_SETTINGS.get(key) is not None}
    if options:
        payload["options"] = options
    response = requests.post(url, json=payload, timeout=OLLAMA_SETTINGS.get("timeout_seconds", 120))
    response.raise_for_status()
    return response.json()["response"]

def llm_classify_feature(feature_name: str, feature_type: str, expected_range: str) -> FeatureCategory:
    """Classify feature into a category using LLM"""
    lang_instruction = language_hint(
        "However, the final output must be exactly one of the category names listed above (in English)."
    )
    prompt = f"""
    Classify this clinical feature into ONE of these categories:
    
    - laboratory_test_biomarker: Lab tests, biomarkers, immunohistochemistry, blood tests, molecular markers, staining results, receptor status
    - demographic_social: Pure demographic (gender, ethnicity, marital status, socioeconomic factors, education)
    - demographic_clinical: Demographic variables with medical significance (age at diagnosis, age at menopause, age at first childbirth, family history of disease)
    - clinical_measurement: Vital signs, physical measurements, clinical scores, BMI, blood pressure, tumor size
    - treatment_procedure: Medications, surgeries, therapies, interventions
    - disease_diagnosis: Disease names, conditions, diagnoses
    - derived_calculated: Calculated scores, indices, risk scores, composite measures
    
    Feature name: "{feature_name}"
    Feature type: "{feature_type}"
    Expected range: "{expected_range}"
    
    {lang_instruction}
    
    IMPORTANT: 
    - If it contains "age_at" with a medical event (menopause, diagnosis, etc.) → demographic_clinical
    - If it's a biomarker or staining result → laboratory_test_biomarker
    
    Output ONLY the category name (e.g., "demographic_clinical"), nothing else.
    """
    response = run_ollama(prompt).strip().lower()
    
    # Map response to enum
    category_map = {
        "laboratory_test_biomarker": FeatureCategory.LAB_TEST,
        "demographic_social": FeatureCategory.DEMOGRAPHIC,
        "demographic_clinical": FeatureCategory.DEMOGRAPHIC_CLINICAL,
        "clinical_measurement": FeatureCategory.CLINICAL_MEASUREMENT,
        "treatment_procedure": FeatureCategory.TREATMENT,
        "disease_diagnosis": FeatureCategory.DISEASE,
        "derived_calculated": FeatureCategory.DERIVED,
    }
    
    for key, value in category_map.items():
        if key in response:
            return value
    
    print(f"  ⚠️  Could not classify, defaulting to UNKNOWN. LLM response: {response}")
    return FeatureCategory.UNKNOWN

def fetch_rxnorm_synonyms(drug_name: str, max_synonyms: int = 12) -> List[str]:
    """Fetch additional drug synonyms from RxNorm."""
    if not drug_name:
        return []

    synonyms: List[str] = []
    try:
        response = requests.get(
            f"{RXNORM_BASE_URL}/rxcui.json",
            params={"name": drug_name, "search": 1},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    ids = data.get("idGroup", {}).get("rxnormId") or []
    if not isinstance(ids, list):
        ids = [ids]

    for rxcui in ids[:3]:
        if not rxcui:
            continue
        try:
            syn_resp = requests.get(
                f"{RXNORM_BASE_URL}/rxcui/{rxcui}/allProperties.json",
                params={"prop": "names"},
                timeout=10,
            )
            syn_resp.raise_for_status()
            syn_data = syn_resp.json()
        except Exception:
            continue

        prop_group = syn_data.get("propConceptGroup", {})
        concepts = prop_group.get("propConcept") or []
        for concept in concepts:
            name = concept.get("propValue")
            if not name:
                continue
            normalized = name.strip()
            if normalized and normalized not in synonyms:
                synonyms.append(normalized)
            if len(synonyms) >= max_synonyms:
                return synonyms

    return synonyms

# =========================================================
# 🔍 ONTOLOGY SEARCH FUNCTIONS WITH RETRY
# =========================================================
def search_umls_with_retry(search_term: str, api_key: str, limit: int = 10, max_retries: int = 3) -> List[Dict]:
    """Search the official UMLS endpoint with verified TLS and timeout retries."""
    url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params={"string": search_term, "apiKey": api_key}, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for result in data['result']['results'][:limit]:
                results.append({
                    'name': result['name'],
                    'cui': result['ui'],
                    'source': 'UMLS',
                    'search_term': search_term
                })
            return results
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  ⏳ Timeout, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  ⚠️  UMLS timeout after {max_retries} attempts for '{search_term}'")
                return []
        except Exception as e:
            print(f"  ⚠️  UMLS request failed ({type(e).__name__})")
            return []
    
    return []

def get_cui_semantic_types_with_retry(cui: str, api_key: str, max_retries: int = 3) -> Dict:
    """Fetch semantic types with retry logic"""
    
    for attempt in range(max_retries):
        try:
            url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{quote(cui, safe='')}"
            response = requests.get(url, params={"apiKey": api_key}, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            result = data.get('result', {})
            
            return {
                'cui': cui,
                'name': result.get('name', ''),
                'semantic_types': result.get('semanticTypes', []),
                'definitions': result.get('definitions', []),
                'atoms': result.get('atomCount', 0),
                'source': 'UMLS'
            }
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  ⏳ Timeout fetching {cui}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ⚠️  Timeout after {max_retries} attempts for {cui}")
                return {'cui': cui, 'error': 'timeout'}
        except Exception as e:
            return {'cui': cui, 'error': f'request_failed ({type(e).__name__})'}
    
    return {'cui': cui, 'error': 'timeout'}

def search_bioportal_with_retry(search_term: str, api_key: str, limit: int = 10, max_retries: int = 3) -> List[Dict]:
    """Search BioPortal API with retry logic"""
    url = "https://data.bioontology.org/search"
    params = {
        'q': search_term,
        'pagesize': limit,
        'suggest': 'true'
    }
    headers = {"Authorization": f"apikey token={api_key}"}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('collection', [])[:limit]:
                result = {
                    'name': item.get('prefLabel', ''),
                    'id': item.get('@id', ''),
                    'ontology': item.get('links', {}).get('ontology', '').split('/')[-1],
                    'source': 'BioPortal',
                    'search_term': search_term
                }
                
                if 'properties' in item:
                    cui = item['properties'].get('cui') or item['properties'].get('UMLS_CUI')
                    if cui:
                        result['cui'] = cui
                
                results.append(result)
            
            return results
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  ⏳ BioPortal timeout, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ⚠️  BioPortal timeout after {max_retries} attempts for '{search_term}'")
                return []
        except Exception as e:
            print(f"  ⚠️  BioPortal request failed ({type(e).__name__})")
            return []
    
    return []

# Aliasing for backward compatibility
search_umls = search_umls_with_retry
get_cui_semantic_types = get_cui_semantic_types_with_retry
search_bioportal = search_bioportal_with_retry

# =========================================================
# 🎯 CUI RELEVANCE SCORING
# =========================================================
def score_cui_relevance(cui_info: Dict, original_feature_name: str, search_term: str) -> float:
    """Score how relevant a CUI is to the original feature (0-1)"""
    score = 0.5  # Base score
    
    cui_name = cui_info.get('name', '').lower()
    feature_lower = original_feature_name.lower().replace('_', ' ')
    search_lower = search_term.lower()
    
    # Higher score if CUI name closely matches feature name
    if feature_lower in cui_name or cui_name in feature_lower:
        score += 0.3
    
    # Higher score if CUI name closely matches search term
    if search_lower in cui_name or cui_name in search_lower:
        score += 0.2
    
    # Prefer certain semantic types for different feature types
    semantic_types = [st.get('name', '') for st in cui_info.get('semantic_types', [])]
    
    # For lab tests: prefer these types
    if any(term in feature_lower for term in ['biomarker', 'test', 'status', 'stain', 'marker']):
        preferred_types = ['Laboratory Procedure', 'Clinical Attribute', 'Diagnostic Procedure', 'Immunologic Factor']
        if any(pt in semantic_types for pt in preferred_types):
            score += 0.2
    
    # For age/demographic: prefer Finding
    if 'age' in feature_lower:
        if 'Finding' in semantic_types or 'Clinical Attribute' in semantic_types:
            score += 0.2
    
    # Penalize very generic or irrelevant concepts
    irrelevant_terms = ['epileptic', 'seizure', 'therapy changed', 'premenopausal', 'menorrhagia']
    if any(term in cui_name for term in irrelevant_terms):
        score -= 0.4
    
    # Deprioritize overly generic single-word concepts
    generic_terms = ['status', 'finding', 'evaluation', 'test', 'change']
    if any(gt in cui_name for gt in generic_terms) and len(cui_name.split()) <= 2:
        score -= 0.2
    
    return min(1.0, max(0.0, score))

# =========================================================
# 🔧 GENERATE COMMON QUERY PATTERNS (FIXED)
# =========================================================
def generate_common_queries(feature_name: str, normalized_name: str, synonyms: List[str], feature_type: str, expected_range: str) -> List[str]:
    """Generate common query patterns users might use for a SINGLE PATIENT"""
    lang_instruction = language_hint("Ensure each query sounds natural for clinicians in this language.")
    prompt = f"""
    Generate 6-8 natural language queries a clinical researcher might use to ask about THIS SPECIFIC PATIENT's feature value.
    
    CRITICAL CONTEXT: We are building a per-patient knowledge graph. Users ask questions about ONE patient at a time, NOT about populations or aggregates.
    
    Feature: {feature_name}
    Normalized name: {normalized_name}
    Type: {feature_type}
    Values: {expected_range}
    Synonyms: {', '.join(synonyms[:3])}
    
    {lang_instruction}
    
    Generate queries in these patterns:
    1. "what is this patient's [feature]?"
    2. "show me the [feature]"
    3. "what was the [feature] for this case?"
    4. "does this patient have [value]?" (if categorical)
    5. "what is the value of [feature]?"
    6. Use different synonyms for variety
    7. Include at least one question about how clinicians document or abbreviate this concept in notes (e.g., "In medical documentation, how do clinicians refer to alcohol use?" or "What abbreviation is used for alcohol consumption?")
    
    ABSOLUTELY DO NOT generate:
    - ❌ "show me all patients with..."
    - ❌ "find patients where..."
    - ❌ "what is the average/median/distribution..."
    - ❌ "how many patients..."
    - ❌ Any population-level or aggregate queries
    
    Output as a JSON array of strings only.
    Example for "ER status": ["what is this patient's ER status?", "show me the estrogen receptor result", "is this patient ER positive?", "what was the ER immunohistochemistry result for this case?", "in medical documentation, how do clinicians refer to estrogen receptor status?"]
    """
    
    output = run_ollama(prompt)
    try:
        # Extract JSON array
        start = output.index('[')
        end = output.rindex(']') + 1
        queries = json.loads(output[start:end])
        
        # Filter out any aggregate queries that slipped through
        filtered = []
        bad_phrases = ['all patients', 'patients with', 'find cases', 'average', 'median', 'distribution', 'proportion', 'how many', 'show me patients']
        for q in queries:
            if not any(phrase in q.lower() for phrase in bad_phrases):
                filtered.append(q)
        
        return filtered[:8]
    except Exception as e:
        print(f"  ⚠️  Query generation failed: {e}")
        # Fallback: generate basic per-patient queries, including documentation shorthand
        queries = [
            f"what is this patient's {normalized_name}",
            f"show me the {normalized_name}",
            f"what was the {feature_name.replace('_', ' ')} for this case",
            f"what is the value of {normalized_name}",
            f"in clinical documentation, how do providers refer to {normalized_name}",
            f"what abbreviation is used in notes for {normalized_name}"
        ]
        
        # Add value-specific queries if categorical
        if expected_range and ',' in expected_range:
            values = [v.strip() for v in expected_range.split(',')][:2]
            for val in values:
                queries.append(f"does this patient have {val.lower()} {normalized_name}")
                queries.append(f"is the {normalized_name} {val.lower()}")
        
        # Add synonym variations
        for syn in synonyms[:2]:
            queries.append(f"what is this patient's {syn}")
        
        return queries[:8]
# =========================================================
# 🔗 IDENTIFY RELATED FEATURES (IMPROVED)
# =========================================================
def identify_related_features(feature_name: str, category: str, normalized_name: str, all_feature_names: List[str]) -> List[str]:
    """Identify potentially related features using LLM"""
    if len(all_feature_names) <= 1:
        return []
    
    lang_instruction = language_hint(
        "Reason in this language, but the JSON array must contain EXACT feature names from the provided list (lowercase snake_case)."
    )
    prompt = f"""
    Given this clinical feature and a list of other features, identify 3-5 features that are clinically or scientifically related.
    Consider:
    - Features commonly measured together (e.g., ER/PR/HER2 in breast cancer)
    - Features in the same clinical domain (e.g., reproductive history variables)
    - Features that inform each other (e.g., age at diagnosis and stage)
    
    Target feature: {feature_name}
    Normalized name: {normalized_name}
    Category: {category}
    
    {lang_instruction}
    
    Available features:
    {json.dumps(all_feature_names[:50], indent=2)}
    
    Output ONLY a JSON array of feature names that are actually related. If none are related, output an empty array [].
    Example: ["er_receptor_status", "pr_receptor_status", "her2_status"]
    """
    
    output = run_ollama(prompt)
    try:
        start = output.index('[')
        end = output.rindex(']') + 1
        related = json.loads(output[start:end])
        # Filter to only include features that actually exist
        return [f for f in related if f in all_feature_names and f != feature_name][:5]
    except:
        return []

# =========================================================
# 📝 GENERATE EHR FREE TEXT EXAMPLES
# =========================================================
def generate_ehr_examples(feature_name: str, normalized_name: str, feature_type: str, expected_range: str, category: str) -> List[str]:
    """Generate realistic EHR free text examples showing how this feature appears in clinical notes"""
    lang_instruction = language_hint(
        "Ensure the EHR snippets reflect how clinicians document in this language, including localized abbreviations."
    )
    prompt = f"""
    Generate 2-3 realistic examples of how this clinical feature would appear in EHR free text notes (progress notes, discharge summaries, pathology reports, etc.).
    
    Feature: {feature_name}
    Normalized name: {normalized_name}
    Category: {category}
    Type: {feature_type}
    Values: {expected_range}
    
    {lang_instruction}
    
    Requirements:
    - Use natural clinical language as doctors actually write
    - Include relevant surrounding context (1-2 sentences)
    - Show different phrasings/contexts
    - Be concise and realistic
    - Include both positive and negative findings if categorical
    - Use medical abbreviations where appropriate
    
    Output as a JSON array of strings, each being a short clinical note excerpt.
    
    Example for "ER receptor status":
    [
      "Pathology: Invasive ductal carcinoma, grade 2. ER positive (90%), PR positive (80%), HER2 negative.",
      "IHC results show strongly positive estrogen receptor staining in tumor cells.",
      "Patient has ER-negative breast cancer, discussed treatment options including chemotherapy."
    ]
    
    Example for "age at menopause onset":
    [
      "PMH significant for early menopause at age 42.",
      "Patient reports natural menopause occurring at approximately 51 years of age.",
      "Reproductive history: G2P2, menopause onset age 48, no HRT."
    ]
    """
    
    output = run_ollama(prompt)
    try:
        start = output.index('[')
        end = output.rindex(']') + 1
        examples = json.loads(output[start:end])
        return examples[:3]
    except Exception as e:
        print(f"  ⚠️  EHR example generation failed: {e}")
        # Fallback examples
        if feature_type == "categorical" and expected_range:
            values = expected_range.split(',')
            return [
                f"{normalized_name}: {values[0].strip()}",
                f"Patient's {normalized_name} is {values[-1].strip()}",
            ]
        else:
            return [
                f"{normalized_name} documented",
                f"Patient has {normalized_name}"
            ]

# =========================================================
# 🔬 ENRICHMENT STRATEGIES BY CATEGORY
# =========================================================

def enrich_lab_test(feature_name: str, feature_type: str, expected_range: str, umls_api_key: str, bioportal_api_key: str) -> Dict:
    """Enrich laboratory test/biomarker features"""
    print("  Strategy: Laboratory Test/Biomarker")
    lang_instruction = language_hint(
        "Provide the description, synonyms, and keywords using terminology commonly used in this language."
    )
    
    prompt = f"""
    You are a clinical laboratory specialist. Generate comprehensive metadata for this lab test/biomarker feature.
    
    Feature name: "{feature_name}"
    Feature type: "{feature_type}"
    Expected values: "{expected_range}"
    
    {lang_instruction}
    
    Output valid JSON with these fields:
    {{
      "normalized_name": "clear, clinical name",
      "description": "what this test measures and its clinical significance (2-3 sentences)",
      "synonyms": ["list of 8-12 alternative names, abbreviations, common phrasings (include standard clinical shorthand/abbreviations when they exist)"],
      "semantic_keywords": ["list of 8-12 related clinical concepts"],
      "clinical_context": "when and why this test is used (2-3 sentences)",
      "test_type": "type of test",
      "search_terms": ["3-5 specific terms for ontology search"],
      "interpretation_guide": "brief guide on interpreting results"
    }}
    """
    
    llm_output = run_ollama(prompt)
    try:
        js, je = llm_output.index("{"), llm_output.rindex("}") + 1
        llm_data = json.loads(llm_output[js:je])
    except Exception as e:
        print(f"  ⚠️  LLM parse error: {e}")
        llm_data = {
            "normalized_name": feature_name.replace("_", " ").title(),
            "description": "",
            "synonyms": [],
            "semantic_keywords": [],
            "clinical_context": "",
            "test_type": "laboratory test",
            "search_terms": [feature_name.replace("_", " ")],
            "interpretation_guide": ""
        }
    
    # Search ontologies
    ontology_results = {"umls": [], "bioportal": []}
    search_terms = llm_data.get("search_terms", [feature_name.replace("_", " ")])
    
    print(f"  Searching ontologies with: {search_terms}")
    for term in search_terms[:3]:
        umls_results = search_umls(term, umls_api_key, limit=8)
        ontology_results["umls"].extend(umls_results)
        
        if bioportal_api_key:
            bp_results = search_bioportal(term, bioportal_api_key, limit=8)
            ontology_results["bioportal"].extend(bp_results)
    
    # Deduplicate by CUI
    seen_cuis = set()
    unique_umls = []
    for r in ontology_results["umls"]:
        if r['cui'] not in seen_cuis:
            seen_cuis.add(r['cui'])
            unique_umls.append(r)
    ontology_results["umls"] = unique_umls
    
    # Generate EHR examples
    print("  Generating EHR free text examples...")
    ehr_examples = generate_ehr_examples(
        feature_name, 
        llm_data.get('normalized_name', feature_name),
        feature_type,
        expected_range,
        FeatureCategory.LAB_TEST.value
    )
    
    return {
        **llm_data,
        "category": FeatureCategory.LAB_TEST.value,
        "ontology_mappings": ontology_results,
        "ehr_examples": ehr_examples
    }

def enrich_demographic(feature_name: str, feature_type: str, expected_range: str) -> Dict:
    """Enrich demographic/social features - LLM only"""
    print("  Strategy: Demographic/Social (LLM-only)")
    lang_instruction = language_hint(
        "Use culturally and linguistically appropriate terminology in this language."
    )
    
    prompt = f"""
    Generate comprehensive metadata for this demographic/social feature.
    
    Feature name: "{feature_name}"
    Feature type: "{feature_type}"
    Expected values: "{expected_range}"
    
    {lang_instruction}
    
    Output valid JSON with:
    {{
      "normalized_name": "clear, readable name",
      "description": "what this demographic variable represents (2-3 sentences)",
      "synonyms": ["8-12 alternative phrasings, abbreviations, and related terms (include common clinical shorthand such as ETOH for alcohol use when applicable)"],
      "semantic_keywords": ["8-12 related demographic/social concepts"],
      "clinical_context": "why this is collected in clinical research (2-3 sentences)",
      "variable_type": "type"
    }}
    """
    
    llm_output = run_ollama(prompt)
    try:
        js, je = llm_output.index("{"), llm_output.rindex("}") + 1
        llm_data = json.loads(llm_output[js:je])
    except:
        llm_data = {
            "normalized_name": feature_name.replace("_", " ").title(),
            "description": "",
            "synonyms": [],
            "semantic_keywords": [],
            "clinical_context": "",
            "variable_type": "demographic variable"
        }
    
    # Generate EHR examples
    print("  Generating EHR free text examples...")
    ehr_examples = generate_ehr_examples(
        feature_name, 
        llm_data.get('normalized_name', feature_name),
        feature_type,
        expected_range,
        FeatureCategory.DEMOGRAPHIC.value
    )
    
    return {
        **llm_data,
        "category": FeatureCategory.DEMOGRAPHIC.value,
        "ontology_mappings": {"umls": [], "bioportal": []},
        "note": "Demographic variables typically don't have standard ontology mappings",
        "ehr_examples": ehr_examples
    }

def enrich_demographic_clinical(feature_name: str, feature_type: str, expected_range: str, umls_api_key: str, bioportal_api_key: str) -> Dict:
    """Enrich demographic features with clinical significance"""
    print("  Strategy: Demographic-Clinical Hybrid (LLM + Ontologies)")
    lang_instruction = language_hint(
        "All narrative descriptions, synonyms, and keywords must be expressed in this language."
    )
    
    prompt = f"""
    Generate comprehensive metadata for this clinically-significant demographic feature.
    
    Feature name: "{feature_name}"
    Feature type: "{feature_type}"
    Expected values: "{expected_range}"
    
    {lang_instruction}
    
    Output valid JSON with:
    {{
      "normalized_name": "clear name",
      "description": "what this represents and clinical significance (2-3 sentences)",
      "synonyms": ["8-12 alternative phrasings including clinical terms and common abbreviations used in documentation"],
      "semantic_keywords": ["8-12 related clinical and demographic concepts"],
      "clinical_context": "medical relevance and research use (2-3 sentences)",
      "variable_type": "type",
      "search_terms": ["2-4 SPECIFIC medical terms for ontology search - use exact clinical phrases"]
    }}
    
    IMPORTANT for search_terms: Be very specific. For "age at menopause", use "age at menopause" or "menopausal age", NOT generic terms like "hormonal changes".
    """
    
    llm_output = run_ollama(prompt)
    try:
        js, je = llm_output.index("{"), llm_output.rindex("}") + 1
        llm_data = json.loads(llm_output[js:je])
    except:
        llm_data = {
            "normalized_name": feature_name.replace("_", " ").title(),
            "synonyms": [],
            "semantic_keywords": [],
            "search_terms": [feature_name.replace("_", " ")]
        }
    
    # Try ontology search with specific terms only
    ontology_results = {"umls": [], "bioportal": []}
    search_terms = llm_data.get("search_terms", [feature_name.replace("_", " ")])
    
    # Filter out generic terms
    filtered_terms = [t for t in search_terms if len(t.split()) >= 2 or len(t) > 10]
    if not filtered_terms:
        filtered_terms = search_terms[:1]
    
    for term in filtered_terms[:2]:
        umls_results = search_umls(term, umls_api_key, limit=5)
        ontology_results["umls"].extend(umls_results)
        
        if bioportal_api_key:
            bp_results = search_bioportal(term, bioportal_api_key, limit=5)
            ontology_results["bioportal"].extend(bp_results)
    
    # Generate EHR examples
    print("  Generating EHR free text examples...")
    ehr_examples = generate_ehr_examples(
        feature_name, 
        llm_data.get('normalized_name', feature_name),
        feature_type,
        expected_range,
        FeatureCategory.DEMOGRAPHIC_CLINICAL.value
    )

    return {
        **llm_data,
        "category": FeatureCategory.DEMOGRAPHIC_CLINICAL.value,
        "ontology_mappings": ontology_results,
        "ehr_examples": ehr_examples
    }


def enrich_treatment(feature_name: str, feature_type: str, expected_range: str, umls_api_key: str, bioportal_api_key: str) -> Dict:
    """Enrich treatment/procedure features (LLM + RxNorm + ontology)."""
    print("  Strategy: Treatment/Procedure (LLM + RxNorm + Ontologies)")
    lang_instruction = language_hint(
        "Describe the therapy using terminology clinicians use in this language. Include drug aliases and regimen names."
    )

    prompt = f"""
    You are an oncology pharmacist. Generate metadata for this treatment or medication feature.

    Feature name: "{feature_name}"
    Feature type: "{feature_type}"
    Expected values: "{expected_range}"

    {lang_instruction}

    Output valid JSON with:
    {{
      "normalized_name": "clinical treatment name",
      "description": "what this therapy is and when it is used (2-3 sentences)",
      "synonyms": ["8-12 alternative names (brand/generic), regimen abbreviations, common misspellings"],
      "semantic_keywords": ["8-12 related treatment concepts"],
      "clinical_context": "how/when this therapy is administered (2-3 sentences)",
      "treatment_type": "drug, regimen, procedure, etc.",
      "search_terms": ["3-5 precise medical search terms"]
    }}
    """

    llm_output = run_ollama(prompt)
    try:
        js, je = llm_output.index("{"), llm_output.rindex("}") + 1
        llm_data = json.loads(llm_output[js:je])
    except Exception as exc:
        print(f"  ⚠️  LLM parse error: {exc}")
        llm_data = {
            "normalized_name": feature_name.replace("_", " ").title(),
            "synonyms": [],
            "semantic_keywords": [],
            "search_terms": [feature_name.replace("_", " ")],
            "clinical_context": "",
            "treatment_type": "treatment",
        }

    rxnorm_synonyms = fetch_rxnorm_synonyms(llm_data.get("normalized_name", feature_name))
    if rxnorm_synonyms:
        print(f"  Added {len(rxnorm_synonyms)} RxNorm synonyms")
        existing_synonyms = llm_data.get("synonyms", [])
        merged = existing_synonyms + rxnorm_synonyms
        llm_data["synonyms"] = list(dict.fromkeys([s for s in merged if s]))

    ontology_results = {"umls": [], "bioportal": []}
    search_terms = llm_data.get("search_terms", [feature_name.replace("_", " ")])
    for term in search_terms[:3]:
        if not term:
            continue
        umls_results = search_umls(term, umls_api_key, limit=6)
        ontology_results["umls"].extend(umls_results)
        if bioportal_api_key:
            bp_results = search_bioportal(term, bioportal_api_key, limit=6)
            ontology_results["bioportal"].extend(bp_results)

    print("  Generating EHR free text examples...")
    ehr_examples = generate_ehr_examples(
        feature_name,
        llm_data.get("normalized_name", feature_name),
        feature_type,
        expected_range,
        FeatureCategory.TREATMENT.value,
    )

    return {
        **llm_data,
        "category": FeatureCategory.TREATMENT.value,
        "ontology_mappings": ontology_results,
        "ehr_examples": ehr_examples,
    }

def enrich_clinical_measurement(feature_name: str, feature_type: str, expected_range: str, umls_api_key: str, bioportal_api_key: str) -> Dict:
    """Enrich clinical measurement features"""
    print("  Strategy: Clinical Measurement (LLM + Ontologies)")
    lang_instruction = language_hint(
        "Descriptions, synonyms, and context must mirror how clinicians document this measurement in this language."
    )
    
    prompt = f"""
    Generate comprehensive metadata for this clinical measurement.
    
    Feature name: "{feature_name}"
    Feature type: "{feature_type}"
    Expected values: "{expected_range}"
    
    {lang_instruction}
    
    Output valid JSON with:
    {{
      "normalized_name": "clear, clinical name",
      "description": "what this measures and clinical significance (2-3 sentences)",
      "synonyms": ["8-12 alternative names and abbreviations (include standard clinical shorthand used in notes)"],
      "semantic_keywords": ["8-12 related clinical concepts"],
      "clinical_context": "when and why measured (2-3 sentences)",
      "measurement_type": "type",
      "search_terms": ["3-5 specific terms for ontology search"]
    }}
    """
    
    llm_output = run_ollama(prompt)
    try:
        js, je = llm_output.index("{"), llm_output.rindex("}") + 1
        llm_data = json.loads(llm_output[js:je])
    except:
        llm_data = {
            "normalized_name": feature_name.replace("_", " ").title(),
            "synonyms": [],
            "semantic_keywords": [],
            "search_terms": [feature_name.replace("_", " ")]
        }
    
    # Search ontologies
    ontology_results = {"umls": [], "bioportal": []}
    search_terms = llm_data.get("search_terms", [feature_name.replace("_", " ")])
    
    for term in search_terms[:3]:
        umls_results = search_umls(term, umls_api_key, limit=5)
        ontology_results["umls"].extend(umls_results)
        
        if bioportal_api_key:
            bp_results = search_bioportal(term, bioportal_api_key, limit=5)
            ontology_results["bioportal"].extend(bp_results)
    
    # Generate EHR examples
    print("  Generating EHR free text examples...")
    ehr_examples = generate_ehr_examples(
        feature_name, 
        llm_data.get('normalized_name', feature_name),
        feature_type,
        expected_range,
        FeatureCategory.CLINICAL_MEASUREMENT.value
    )
    
    return {
        **llm_data,
        "category": FeatureCategory.CLINICAL_MEASUREMENT.value,
        "ontology_mappings": ontology_results,
        "ehr_examples": ehr_examples
    }

def enrich_disease(feature_name: str, feature_type: str, expected_range: str, umls_api_key: str, bioportal_api_key: str) -> Dict:
    """Enrich disease/diagnosis features"""
    print("  Strategy: Disease/Diagnosis (Ontologies + LLM)")
    lang_instruction = language_hint(
        "Provide the narrative description, synonyms, and keywords in this language using standard clinical phrasing."
    )
    
    # For diseases, prioritize ontology search
    search_terms = [
        feature_name.replace("_", " "),
        feature_name.replace("_diagnosis", "").replace("_", " "),
        feature_name.replace("_status", "").replace("_", " ")
    ]
    
    ontology_results = {"umls": [], "bioportal": []}
    
    for term in search_terms[:2]:
        umls_results = search_umls(term, umls_api_key, limit=10)
        ontology_results["umls"].extend(umls_results)
        
        if bioportal_api_key:
            bp_results = search_bioportal(term, bioportal_api_key, limit=10)
            ontology_results["bioportal"].extend(bp_results)
    
    # LLM enrichment
    prompt = f"""
    Generate clinical metadata for this disease/diagnosis feature.
    
    Feature name: "{feature_name}"
    Expected values: "{expected_range}"
    
    {lang_instruction}
    
    Output valid JSON with:
    {{
      "normalized_name": "clinical disease name",
      "description": "clinical description (2-3 sentences)",
      "synonyms": ["8-12 alternative names and abbreviations (include standard shorthand used in clinical documentation)"],
      "semantic_keywords": ["8-12 related clinical terms"],
      "clinical_context": "clinical significance (2-3 sentences)"
    }}
    """
    
    llm_output = run_ollama(prompt)
    try:
        js, je = llm_output.index("{"), llm_output.rindex("}") + 1
        llm_data = json.loads(llm_output[js:je])
    except:
        llm_data = {
            "normalized_name": feature_name.replace("_", " ").title(),
            "synonyms": [],
            "semantic_keywords": []
        }
    
    # Generate EHR examples
    print("  Generating EHR free text examples...")
    ehr_examples = generate_ehr_examples(
        feature_name, 
        llm_data.get('normalized_name', feature_name),
        feature_type,
        expected_range,
        FeatureCategory.DISEASE.value
    )
    
    return {
        **llm_data,
        "category": FeatureCategory.DISEASE.value,
        "ontology_mappings": ontology_results,
        "ehr_examples": ehr_examples
    }

def enrich_derived(feature_name: str, feature_type: str, expected_range: str) -> Dict:
    """Enrich derived/calculated features"""
    print("  Strategy: Derived/Calculated (LLM-only)")
    lang_instruction = language_hint(
        "Express the description, synonyms, and clinical context entirely in this language."
    )
    
    prompt = f"""
    Generate metadata for this derived/calculated clinical feature.
    
    Feature name: "{feature_name}"
    Feature type: "{feature_type}"
    Expected values: "{expected_range}"
    
    {lang_instruction}
    
    Output valid JSON with:
    {{
      "normalized_name": "clear name",
      "description": "what is calculated and what it represents (2-3 sentences)",
      "synonyms": ["8-12 alternative names and common abbreviations if they exist"],
      "semantic_keywords": ["8-12 related concepts"],
      "clinical_context": "how this is used clinically (2-3 sentences)",
      "calculation_note": "brief note about calculation if known"
    }}
    """
    
    llm_output = run_ollama(prompt)
    try:
        js, je = llm_output.index("{"), llm_output.rindex("}") + 1
        llm_data = json.loads(llm_output[js:je])
    except:
        llm_data = {
            "normalized_name": feature_name.replace("_", " ").title(),
            "synonyms": [],
            "semantic_keywords": []
        }
    
    # Generate EHR examples
    print("  Generating EHR free text examples...")
    ehr_examples = generate_ehr_examples(
        feature_name, 
        llm_data.get('normalized_name', feature_name),
        feature_type,
        expected_range,
        FeatureCategory.DERIVED.value
    )
    
    return {
        **llm_data,
        "category": FeatureCategory.DERIVED.value,
        "ontology_mappings": {"umls": [], "bioportal": []},
        "note": "Derived variables typically don't have standard ontology mappings",
        "ehr_examples": ehr_examples
    }


# =========================================================
# 🎯 MAIN PROCESSING FUNCTION
# =========================================================
def process_features_with_ontology_mapping(
    features_file: str = str(Path(__file__).resolve().parent / "config" / "feature_specs_ricci.yaml"),
    output_dir: str = "oncoraggraph/config",
    output_file: str = "feature_ontology_mappings.json",
    context: str = "Oncology clinical research dataset",
    language: str = "english",
    host: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    timeout_seconds: int = 120,
    max_concepts: int = 5,
    min_relevance: float = 0.6,
) -> List[Dict]:
    """
    Process features from a YAML file and map them to ontologies using classification-based routing.
    """
    features = load_feature_specs(features_file)
    if type(max_concepts) is not int or max_concepts < 1:
        raise ValueError("max_concepts must be a positive integer")
    if isinstance(min_relevance, bool) or not isinstance(min_relevance, (int, float)) or not 0 <= min_relevance <= 1:
        raise ValueError("min_relevance must be between zero and one")
    if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0):
        raise ValueError("temperature must be finite and nonnegative")
    for key, value in (("host", host), ("model", model)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a nonempty string")
    if num_ctx is not None and (type(num_ctx) is not int or num_ctx < 1):
        raise ValueError("num_ctx must be a positive integer")
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError("timeout_seconds must be a positive integer")
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer")
    OLLAMA_SETTINGS.clear()
    OLLAMA_SETTINGS.update(host=host, model=model, temperature=temperature, num_ctx=num_ctx, seed=seed, timeout_seconds=timeout_seconds)
    set_target_language(language)
    print(f"🌐 Target language: {TARGET_LANGUAGE}")
    # Load API keys
    umls_api_key = os.getenv('UMLS_API_KEY')
    bioportal_api_key = os.getenv('BIOPORTAL_API_KEY')
    
    if not umls_api_key:
        raise ValueError("UMLS_API_KEY not found in .env file")
    _load_wordnet()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_path.absolute()}")
    
    # Load features from YAML file
    print(f"📂 Loading features from: {features_file}")
    all_feature_names = [f.get('name', '') for f in features]
    
    print(f"\n{'='*80}")
    print(f"🎯 Processing {len(features)} features")
    print(f"{'='*80}")
    
    # Process each feature
    all_feature_results = []
    
    for i, feature in enumerate(features, 1):
        feature_name = feature.get('name', '')
        feature_type = feature.get('type', '')
        expected_range_raw = feature.get('expected_range', '')
        if isinstance(expected_range_raw, list):
            expected_range_list = [str(v).strip() for v in expected_range_raw if str(v).strip()]
            expected_range = ", ".join(expected_range_list)
        elif isinstance(expected_range_raw, str):
            expected_range = expected_range_raw
            expected_range_list = [v.strip() for v in expected_range_raw.split(",") if v.strip()]
        else:
            expected_range = str(expected_range_raw)
            expected_range_list = [expected_range] if expected_range else []
        
        print(f"\n{'='*80}")
        print(f"🔍 [{i}/{len(features)}] Processing: {feature_name}")
        print(f"{'='*80}")
        
        # Step 1: Classify the feature
        print("\n📋 Step 1: Classifying feature...")
        category = llm_classify_feature(feature_name, feature_type, expected_range)
        print(f"  Category: {category.value}")
        
        # Step 2: Route to appropriate enrichment strategy
        print("\n🔧 Step 2: Enriching with category-specific strategy...")
        
        enrichment = {}
        if category == FeatureCategory.LAB_TEST:
            enrichment = enrich_lab_test(feature_name, feature_type, expected_range, umls_api_key, bioportal_api_key)
        elif category == FeatureCategory.DEMOGRAPHIC:
            enrichment = enrich_demographic(feature_name, feature_type, expected_range)
        elif category == FeatureCategory.DEMOGRAPHIC_CLINICAL:
            enrichment = enrich_demographic_clinical(feature_name, feature_type, expected_range, umls_api_key, bioportal_api_key)
        elif category == FeatureCategory.CLINICAL_MEASUREMENT:
            enrichment = enrich_clinical_measurement(feature_name, feature_type, expected_range, umls_api_key, bioportal_api_key)
        elif category == FeatureCategory.TREATMENT:
            enrichment = enrich_treatment(feature_name, feature_type, expected_range, umls_api_key, bioportal_api_key)
        elif category == FeatureCategory.DISEASE:
            enrichment = enrich_disease(feature_name, feature_type, expected_range, umls_api_key, bioportal_api_key)
        elif category == FeatureCategory.DERIVED:
            enrichment = enrich_derived(feature_name, feature_type, expected_range)
        else:
            # Fallback for UNKNOWN
            print("  ⚠️  Category unknown, using clinical measurement strategy")
            enrichment = enrich_clinical_measurement(feature_name, feature_type, expected_range, umls_api_key, bioportal_api_key)
        
        # Step 3: Verify semantic types for CUIs and score relevance
        print("\n🔬 Step 3: Verifying semantic types and scoring relevance...")
        all_cuis = set()
        
        ontology_mappings = enrichment.get('ontology_mappings', {})
        for source in ontology_mappings:
            for res in ontology_mappings[source]:
                cui = res.get('cui')
                if cui:
                    all_cuis.add(cui)
        
        print(f"  Found {len(all_cuis)} unique CUIs")
        
        semantic_types = {}
        cui_relevance_scores = {}
        
        for cui in all_cuis:
            semantic_info = get_cui_semantic_types(cui, umls_api_key)
            semantic_types[cui] = semantic_info
            
            # Score relevance
            search_term = ""
            for source in ontology_mappings:
                for res in ontology_mappings[source]:
                    if res.get('cui') == cui:
                        search_term = res.get('search_term', '')
                        break
            
            relevance_score = score_cui_relevance(semantic_info, feature_name, search_term)
            cui_relevance_scores[cui] = relevance_score
            
            if 'semantic_types' in semantic_info and semantic_info['semantic_types']:
                sem_types = [st.get('name', st) for st in semantic_info['semantic_types'][:2]]
                print(f"  {cui}: {semantic_info.get('name', '')} ({', '.join(sem_types)}) [score: {relevance_score:.2f}]")
        
        # Sort CUIs by relevance and keep top ones
        sorted_cuis = sorted(cui_relevance_scores.items(), key=lambda x: x[1], reverse=True)
        top_cuis = [cui for cui, score in sorted_cuis if score >= min_relevance][:max_concepts]
        
        if top_cuis:
            print(f"  ✅ Top relevant CUIs: {', '.join(top_cuis)}")
        else:
            print(f"  ⚠️  No highly relevant CUIs found (all scores < {min_relevance})")
        
        # Step 4: Generate common query patterns
        print("\n💬 Step 4: Generating common query patterns...")
        common_queries = generate_common_queries(
            feature_name, 
            enrichment.get('normalized_name', ''),
            enrichment.get('synonyms', []),
            feature_type,
            expected_range
        )
        print(f"  Generated {len(common_queries)} query patterns")
        for q in common_queries[:3]:
            print(f"    - {q}")
        
        # Step 5: Identify related features
        print("\n🔗 Step 5: Identifying related features...")
        related_features = identify_related_features(
            feature_name, 
            category.value,
            enrichment.get('normalized_name', ''),
            all_feature_names
        )
        if related_features:
            print(f"  Found {len(related_features)} related features: {', '.join(related_features)}")
        else:
            print(f"  No strongly related features identified")
        
        feature_entry = dict(feature)
        feature_entry.setdefault("medical_context", context)

        output_format = None
        expected_values: list[str] = []
        if str(feature_type).lower() == "categorical" and expected_range_list:
            values = expected_range_list
            expected_values = expected_range_list
            options = {}
            for idx, value in enumerate(values):
                key = chr(ord("A") + idx)
                options[key] = value
            missing_key = "C"
            used_keys = set(options.keys())
            if missing_key in used_keys:
                next_ord = ord("A")
                while chr(next_ord) in used_keys:
                    next_ord += 1
                missing_key = chr(next_ord)
            options[missing_key] = "Missing"
            output_format = {
                "type": "categorical",
                "options": options
            }

        # Combine all data
        result = {
            'feature': feature_entry,
            'category': category.value,
            'enrichment': enrichment,
            'semantic_types': semantic_types,
            'cui_relevance_scores': cui_relevance_scores,
            'top_cuis': top_cuis,
            'common_queries': common_queries,
            'related_features': related_features
        }

        numeric_types = {"numeric", "number", "float", "integer", "int", "decimal"}
        if str(feature_type).strip().lower() in numeric_types:
            guidelines = [
                "Only return a numeric value when the context explicitly states it (e.g., 'Age 28 at birth of first child').",
                "Do not infer or estimate the value by calculating from related ages, dates, or other indirect information.",
                'If the context lacks an explicit statement of the value, respond with "Missing".'
            ]
            rules_block = result.setdefault("rules", {})
            existing_guidelines = rules_block.get("extraction_guidelines") or []
            # Preserve any existing guidance while ensuring deterministic order
            combined = []
            for entry in existing_guidelines:
                if entry and entry not in combined:
                    combined.append(entry)
            for entry in guidelines:
                if entry not in combined:
                    combined.append(entry)
            if combined:
                rules_block["extraction_guidelines"] = combined

        rules_block = result.setdefault("rules", {})
        keyword_set: set[str] = set()

        def _normalize(text: str) -> str:
            return re.sub(r"\s+", " ", text.replace("_", " ").strip().lower())

        raw_phrases = [_normalize(feature_name)]
        for group in (
            enrichment.get("synonyms", []),
            enrichment.get("semantic_keywords", []),
            enrichment.get("search_terms", []),
        ):
            for item in group or []:
                if item:
                    raw_phrases.append(_normalize(item))

        def _variants(phrase: str) -> set[str]:
            variants = {phrase}
            cleaned = phrase.replace("-", " ")
            variants.add(cleaned)
            if "menopause" in cleaned:
                variants.add(cleaned.replace("menopause", "menopausal"))
            if "menopausal" in cleaned:
                variants.add(cleaned.replace("menopausal", "menopause"))
            keep: set[str] = set()
            for v in variants:
                stripped = v.replace(" ", "")
                if len(stripped) >= 4 or any(ch.isdigit() for ch in stripped):
                    keep.add(v)
            return keep

        base_tokens: set[str] = set()
        for phrase in raw_phrases:
            if not phrase:
                continue
            keyword_set.update(_variants(phrase))
            for token in phrase.split():
                cleaned = re.sub(r"[^a-z0-9]", "", token)
                if not cleaned:
                    continue
                if len(cleaned) >= 4 or any(ch.isdigit() for ch in cleaned):
                    base_tokens.add(cleaned)

        # WordNet expansion for synonym discovery (required)
        # Critical for handling clinical terminology variants (e.g., menopause/menopausal)
        extra = set()
        for token in list(base_tokens):
            if any(ch.isdigit() for ch in token):
                continue  # WordNet won't cover alphanumeric assay names like AE1
            try:
                synsets = wn.synsets(token)
            except Exception:
                continue
            for syn in synsets:
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ").lower()
                    if len(name) >= 5:
                        extra.update(_variants(name))
                if len(extra) > 50:
                    break
            if len(extra) > 50:
                break
        keyword_set.update(extra)

        keyword_set.update(base_tokens)

        if expected_range_list:
            for value in expected_range_list:
                cleaned_value = value.strip()
                if not cleaned_value:
                    continue
                normalized_value = _normalize(cleaned_value)
                keyword_set.update(_variants(normalized_value))
                for token in normalized_value.split():
                    stripped = re.sub(r"[^a-z0-9]", "", token)
                    if stripped:
                        base_tokens.add(stripped)
                canonical = cleaned_value.lower()
                lang_synonyms = EXPECTED_VALUE_SYNONYMS.get(canonical, {})
                for synonym in lang_synonyms.get(TARGET_LANGUAGE, []):
                    normalized_synonym = _normalize(synonym)
                    keyword_set.update(_variants(normalized_synonym))
                    for token in normalized_synonym.split():
                        stripped = re.sub(r"[^a-z0-9]", "", token)
                    if stripped:
                        base_tokens.add(stripped)

        # Cohort-specific heuristics
        feature_name_lower = feature_name.lower()
        # Date features
        if str(feature_type).lower() == "date":
            keyword_set.update(DATE_KEYWORDS.get("birth", []) if "birth" in feature_name_lower else [])
            if "diagnosis" in feature_name_lower:
                keyword_set.update(DATE_KEYWORDS.get("diagnosis", []))
            if "progression" in feature_name_lower:
                keyword_set.update(DATE_KEYWORDS.get("progression", []))
            if "death" in feature_name_lower:
                keyword_set.update(DATE_KEYWORDS.get("death", []))
            if "carbon" in feature_name_lower or "rt" in feature_name_lower:
                keyword_set.update(DATE_KEYWORDS.get("carbon_ion_rt", []))
            if "necrosis" in feature_name_lower:
                keyword_set.update(DATE_KEYWORDS.get("radiation_necrosis", []))
            if "resection" in feature_name_lower and "first" in feature_name_lower:
                keyword_set.update(DATE_KEYWORDS.get("resection_repeat", []))
            elif "resection" in feature_name_lower:
                keyword_set.update(DATE_KEYWORDS.get("resection_initial", []))
            if "extraction_guidelines" in rules_block:
                guidelines = rules_block.get("extraction_guidelines") or []
            else:
                guidelines = []
            guidelines.append("Return the explicit date in the notes (YYYY-MM-DD if possible); otherwise Missing.")
            rules_block["extraction_guidelines"] = sorted({g for g in guidelines if g})

        # Laterality
        if "laterality" in feature_name_lower:
            keyword_set.update(LATERALITY_TERMS)
            guidelines = rules_block.get("extraction_guidelines") or []
            guidelines.append("Return left/right/bilateral/central only if explicitly stated; otherwise Missing.")
            rules_block["extraction_guidelines"] = sorted({g for g in guidelines if g})

        # Topography
        if "topography" in feature_name_lower:
            keyword_set.update(TOPOGRAPHY_TERMS)
            guidelines = rules_block.get("extraction_guidelines") or []
            guidelines.append("Select the code/region explicitly stated for the lesion; if multiple, prefer the one tagged as initial/current; otherwise Missing.")
            rules_block["extraction_guidelines"] = sorted({g for g in guidelines if g})

        # RT doses
        if "dose" in feature_name_lower and "rt" in feature_name_lower:
            keyword_set.update(RT_DOSE_TERMS)
            guidelines = rules_block.get("extraction_guidelines") or []
            guidelines.append("Extract the numeric radiation dose in Gy when stated near Gy/Gray; ignore other numbers; if absent, respond Missing.")
            rules_block["extraction_guidelines"] = sorted({g for g in guidelines if g})

        # Medications/regimens
        med_terms = set()
        for key, syns in MED_SYNONYM_SEEDS.items():
            if key in feature_name_lower:
                med_terms.update(syns)
        if med_terms:
            keyword_set.update(med_terms)
            guidelines = rules_block.get("extraction_guidelines") or []
            guidelines.append("Return Yes if the drug/regimen is mentioned; No only if explicitly negated; otherwise Missing.")
            rules_block["extraction_guidelines"] = sorted({g for g in guidelines if g})

        deduped: list[str] = []
        for term in sorted(keyword_set):
            if not term:
                continue
            term_lower = term.lower()
            if base_tokens and not any(bt in term_lower for bt in base_tokens):
                continue
            if any(difflib.SequenceMatcher(None, term_lower, existing).ratio() >= 0.85 for existing in deduped):
                continue
            deduped.append(term_lower)

        rules_block["keywords"] = sorted(set(deduped))

        if output_format:
            result['output_format'] = output_format

        # Automatically seed post-processing metadata for categorical features.
        if str(feature_type).strip().lower() == "categorical":
            post_cfg: Dict[str, Any] = {}

            # Base terms: reuse the canonical tokens gathered for keywords.
            base_terms = sorted({token for token in base_tokens if len(token) >= 3})
            if not base_terms:
                fallback_tokens = re.findall(r"[a-z0-9]+", feature_name.lower())
                base_terms = sorted({token for token in fallback_tokens if len(token) >= 3})
            if base_terms:
                post_cfg["base_terms"] = base_terms

            # Simple unit detection from keywords (helps frequency heuristics).
            unit_candidates = set()
            unit_lexicon = {
                "drink",
                "drinks",
                "glass",
                "glasses",
                "bottle",
                "bottles",
                "pack",
                "packs",
                "pack-year",
                "pack years",
                "cigarette",
                "cigarettes",
                "pipette",
                "dose",
                "doses",
                "puff",
                "puffs",
                "per day",
                "per week",
                "per month",
                "per year",
            }
            for kw in rules_block.get("keywords", []):
                kw_lower = kw.lower()
                for term in unit_lexicon:
                    if term in kw_lower:
                        unit_candidates.add(term)
            if unit_candidates:
                post_cfg["units"] = sorted(unit_candidates)

            # Category-specific cues seeded from expected values.
            indicators: Dict[str, Dict[str, Any]] = {}
            for label in expected_values:
                normalized_label = label.strip()
                if not normalized_label:
                    continue
                lowered = normalized_label.lower()
                if lowered == "missing":
                    continue
                phrases = set()
                phrases.add(normalized_label)
                phrases.add(lowered)
                cleaned = re.sub(r"[_/-]+", " ", lowered).strip()
                if cleaned:
                    phrases.add(cleaned)
                token_fragments = [
                    frag for frag in re.split(r"[\\s,;/]+", lowered) if len(frag) >= 3
                ]
                for frag in token_fragments:
                    phrases.add(frag)
                if phrases:
                    indicators[lowered] = {"phrases": sorted(phrases)}

            if indicators:
                post_cfg["category_indicators"] = indicators

            if post_cfg:
                result["postprocessing"] = post_cfg
        
        all_feature_results.append(result)
        
        # Save individual feature file
        feature_filename = output_path / f"{feature_name}.json"
        with open(feature_filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Saved to: {feature_filename}")
        
        # Print summary for this feature
        print(f"\n✅ Enrichment complete:")
        print(f"  Normalized Name: {enrichment.get('normalized_name', 'N/A')}")
        print(f"  Synonyms: {len(enrichment.get('synonyms', []))}")
        print(f"  Semantic Keywords: {len(enrichment.get('semantic_keywords', []))}")
        print(f"  Common Queries: {len(common_queries)}")
        print(f"  Top CUIs: {len(top_cuis)}")
        print(f"  Related Features: {len(related_features)}")
        print(f"  EHR Examples: {len(enrichment.get('ehr_examples', []))}")
        if enrichment.get('ehr_examples'):
            print(f"    Sample: {enrichment['ehr_examples'][0][:80]}...")
    
    # Save consolidated results
    print(f"\n{'='*80}")
    print(f"💾 Saving consolidated results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(all_feature_results, f, indent=2)
    
    print(f"✅ All results saved successfully!")
    print(f"{'='*80}")
    
    # Print final summary
    print("\n📊 FINAL SUMMARY:")
    category_counts = {}
    for result in all_feature_results:
        cat = result['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nFeatures by category:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}")
    
    print(f"\n📁 Individual feature files saved in: {output_path.absolute()}")
    print(f"📄 Consolidated file: {output_file}")
    
    print("\nDetailed results:")
    for result in all_feature_results:
        feature_name = result['feature']['name']
        category = result['category']
        enrichment = result['enrichment']
        
        print(f"\n{feature_name} [{category}]:")
        print(f"  → {enrichment.get('normalized_name', 'N/A')}")
        print(f"  Synonyms: {len(enrichment.get('synonyms', []))}")
        print(f"  Top CUIs: {', '.join(result.get('top_cuis', [])[:3])}")
        print(f"  Related: {', '.join(result.get('related_features', [])[:3])}")
    
    return all_feature_results


# =========================================================
# 🚀 MAIN CLI
# =========================================================
def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate feature configs with ontology mappings.")
    parser.add_argument(
        "--features-file",
        default=str(Path(__file__).resolve().parent / "config" / "feature_specs_ricci.yaml"),
        help="Path to the feature specs YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        default="feature_configs_ricci",
        help="Directory to store generated feature configs.",
    )
    parser.add_argument(
        "--output-file",
        default="feature_ontology_mappings_ricci.json",
        help="Filename for the consolidated ontology mappings JSON.",
    )
    parser.add_argument(
        "--context",
        default="Oncology clinical research dataset",
        help="Context string passed to the LLM for generation hints.",
    )
    parser.add_argument(
        "--language",
        default="german",
        help="Target language for generated metadata (e.g., german, english).",
    )
    parser.add_argument("--mode", choices=["ontology", "manual"], default="ontology", help="Ontology enrichment or deterministic configs from user-supplied terms.")
    parser.add_argument("--host", help="Ollama URL; defaults to OLLAMA_HOST.")
    parser.add_argument("--model", help="Ollama model; defaults to OLLAMA_MODEL.")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--num-ctx", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-concepts", type=int, default=5)
    parser.add_argument("--min-relevance", type=float, default=0.6)

    args = parser.parse_args(argv)

    if args.mode == "manual":
        paths = generate_feature_configs(load_feature_specs(args.features_file), args.output_dir, args.language)
        results = [json.loads(path.read_text(encoding="utf-8")) for path in paths.values()]
        Path(args.output_file).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Generated {len(paths)} manual feature configs in {args.output_dir}")
    else:
        process_features_with_ontology_mapping(
            features_file=args.features_file,
            output_dir=args.output_dir,
            output_file=args.output_file,
            context=args.context,
            language=args.language,
            host=args.host,
            model=args.model,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            seed=args.seed,
            timeout_seconds=args.timeout_seconds,
            max_concepts=args.max_concepts,
            min_relevance=args.min_relevance,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
