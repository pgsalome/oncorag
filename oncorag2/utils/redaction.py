"""
Name redaction for patient privacy.

This module provides functions to redact personally identifiable information
from medical documents to maintain patient privacy.
"""

import re
import logging
from typing import List, Set

logger = logging.getLogger(__name__)

# Try to load spaCy for NER
try:
    import spacy

    _HAS_SPACY = True
    try:
        _nlp = spacy.load("en_core_web_sm")
    except:
        _nlp = None
        _HAS_SPACY = False
except ImportError:
    _HAS_SPACY = False
    _nlp = None


def _download_spacy_model() -> bool:
    """
    Download the spaCy model for NER.

    Returns:
        True if successful, False otherwise
    """
    try:
        import subprocess
        logger.info("Downloading spaCy model for name detection...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)

        # Try loading again
        import spacy
        global _nlp, _HAS_SPACY
        _nlp = spacy.load("en_core_web_sm")
        _HAS_SPACY = True
        return True
    except Exception as e:
        logger.error(f"Error downloading spaCy model: {e}")
        return False


def _extract_names_with_regex(text: str) -> Set[str]:
    """
    Extract potential names from text using regex patterns.

    Args:
        text: The text to extract names from

    Returns:
        Set of potential names
    """
    # Common patterns for names in medical documents
    name_patterns = [
        # Pattern for "Name: John Smith" or "Patient Name: John Smith"
        r'(?i)(patient\s+)?name\s*[:]\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
        # Pattern for "Dr. Smith" or similar
        r'(?i)(dr|doctor|md|physician|nurse|rn|pa|np)\s*\.?\s*([A-Z][a-z]+)',
        # Pattern for "Smith, John" format (last name first)
        r'([A-Z][a-z]+),\s*([A-Z][a-z]+)',
        # Simple pattern for potential names (two capitalized words in sequence)
        r'\b([A-Z][a-z]{1,20})\s+([A-Z][a-z]{1,20})\b'
    ]

    # Extract potential names using regex
    potential_names = set()

    for pattern in name_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            if len(match.groups()) >= 1:
                for group in match.groups()[1:]:  # Skip the first group as it's usually a title
                    if group and len(group) > 2:  # Avoid very short matches
                        potential_names.add(group)

    return potential_names


def _extract_names_with_spacy(text: str) -> Set[str]:
    """
    Extract names from text using spaCy NER.

    Args:
        text: The text to extract names from

    Returns:
        Set of names extracted with spaCy
    """
    global _nlp, _HAS_SPACY

    # If spaCy is not available or model is not loaded, try to download
    if not _HAS_SPACY or _nlp is None:
        if not _download_spacy_model():
            return set()

    # Extract person names using spaCy NER
    potential_names = set()
    doc = _nlp(text)

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            potential_names.add(ent.text)

    return potential_names


def redact_names_from_markdown(markdown_text: str) -> str:
    """
    Redact names from markdown text using regex patterns and NER.

    Args:
        markdown_text: The markdown text to redact names from

    Returns:
        Markdown text with names redacted
    """
    # Extract potential names using both methods
    potential_names = _extract_names_with_regex(markdown_text)

    # Use spaCy if available
    if _HAS_SPACY and _nlp is not None:
        potential_names.update(_extract_names_with_spacy(markdown_text))

    # Sort names by length (longest first) to avoid partial replacements
    sorted_names = sorted(potential_names, key=len, reverse=True)

    # Replace each identified name with [REDACTED]
    redacted_text = markdown_text
    for name in sorted_names:
        # Ensure we're replacing complete words by using word boundaries
        pattern = r'\b' + re.escape(name) + r'\b'
        redacted_text = re.sub(pattern, '[REDACTED]', redacted_text)

    return redacted_text