"""PHI removal utilities for clinical text."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

from ..config.system_config import get_system_config


@dataclass
class PHIRemovalResult:
    """Result of PHI removal operation."""
    cleaned_text: str
    original_text: str
    replacements: Dict[str, str]
    removed_count: int
    removed_types: List[str]


class PHIRemover:
    """PHI removal utility class."""
    
    def __init__(self):
        self.config = get_system_config()
        self.patterns = self.config.get_phi_removal_patterns()
        self.replacement = self.config.get_phi_replacement()
    
    def remove_phi_from_text(self, text: str, patterns_to_use: List[str] = None) -> PHIRemovalResult:
        """
        Remove PHI from clinical text.
        
        Args:
            text: Input clinical text
            patterns_to_use: Specific patterns to use (if None, uses all patterns)
        
        Returns:
            PHIRemovalResult with cleaned text and metadata
        """
        if not text or not text.strip():
            return PHIRemovalResult(
                cleaned_text=text,
                original_text=text,
                replacements={},
                removed_count=0,
                removed_types=[]
            )
        
        cleaned_text = text
        replacements = {}
        removed_types = []
        
        # Use specified patterns or all patterns
        patterns = patterns_to_use or list(self.patterns.keys())
        
        for pattern_name in patterns:
            if pattern_name not in self.patterns:
                continue
                
            pattern = self.patterns[pattern_name]
            matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
            
            if matches:
                removed_types.append(pattern_name)
                for match in matches:
                    if isinstance(match, tuple):
                        # Handle capture groups
                        match = match[0] if match[0] else match[1]
                    
                    replacement = f"[{pattern_name.upper()}_REDACTED]"
                    cleaned_text = cleaned_text.replace(match, replacement)
                    replacements[match] = replacement
        
        return PHIRemovalResult(
            cleaned_text=cleaned_text,
            original_text=text,
            replacements=replacements,
            removed_count=len(replacements),
            removed_types=removed_types
        )
    
    def remove_phi_from_context(self, context: str) -> PHIRemovalResult:
        """
        Remove PHI from clinical context before sending to LLM.
        Uses patterns specified in the current LLM configuration.
        """
        llm_config = self.config.get_llm_config()
        patterns_to_use = llm_config.phi_removal_patterns
        
        return self.remove_phi_from_text(context, patterns_to_use)
    
    def should_remove_phi(self) -> bool:
        """Check if PHI removal is required for current LLM backend."""
        return self.config.should_remove_phi()


def remove_phi_from_text(text: str, patterns_to_use: List[str] = None) -> PHIRemovalResult:
    """
    Convenience function to remove PHI from text.
    
    Args:
        text: Input clinical text
        patterns_to_use: Specific patterns to use (if None, uses all patterns)
    
    Returns:
        PHIRemovalResult with cleaned text and metadata
    """
    remover = PHIRemover()
    return remover.remove_phi_from_text(text, patterns_to_use)


def remove_phi_from_context(context: str) -> PHIRemovalResult:
    """
    Convenience function to remove PHI from context before LLM processing.
    
    Args:
        context: Clinical context to clean
    
    Returns:
        PHIRemovalResult with cleaned context and metadata
    """
    remover = PHIRemover()
    return remover.remove_phi_from_context(context)


def should_remove_phi() -> bool:
    """Convenience function to check if PHI removal is needed."""
    remover = PHIRemover()
    return remover.should_remove_phi()


# Advanced PHI removal patterns for clinical text
CLINICAL_PHI_PATTERNS = {
    # Patient identifiers
    "patient_id": r'\b(?:Patient ID|Pt ID|ID):\s*[A-Za-z0-9-]+\b',
    "mrn": r'\b(?:MRN|Medical Record Number|Record #):\s*\d+\b',
    "account_number": r'\b(?:Account|Acct) #:\s*\d+\b',
    
    # Personal information
    "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
    "phone": r'\b(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    
    # Addresses
    "street_address": r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Circle|Cir)\b',
    "zip_code": r'\b\d{5}(?:-\d{4})?\b',
    "city_state": r'\b[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}\b',
    
    # Names (basic patterns - be careful with false positives)
    "patient_name": r'\b(?:Patient|Pt):\s*[A-Z][a-z]+\s+[A-Z][a-z]+\b',
    "doctor_name": r'\b(?:Dr\.|Doctor)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
    
    # Dates that might be sensitive
    "dob": r'\b(?:DOB|Date of Birth|Born):\s*\d{1,2}/\d{1,2}/\d{4}\b',
    "admission_date": r'\b(?:Admitted|Admission):\s*\d{1,2}/\d{1,2}/\d{4}\b',
    
    # Financial information
    "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    "insurance_id": r'\b(?:Insurance|Ins) ID:\s*[A-Za-z0-9-]+\b',
    
    # Clinical identifiers
    "case_number": r'\b(?:Case|Case #):\s*[A-Za-z0-9-]+\b',
    "specimen_id": r'\b(?:Specimen|Spec) ID:\s*[A-Za-z0-9-]+\b',
}


def get_clinical_phi_patterns() -> Dict[str, str]:
    """Get comprehensive clinical PHI patterns."""
    return CLINICAL_PHI_PATTERNS.copy()


__all__ = [
    "PHIRemover",
    "PHIRemovalResult", 
    "remove_phi_from_text",
    "remove_phi_from_context",
    "should_remove_phi",
    "get_clinical_phi_patterns"
]
