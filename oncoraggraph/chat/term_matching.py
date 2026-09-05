"""Utilities for typo-tolerant term matching."""

from __future__ import annotations

import difflib
import re
from typing import Iterable, List, Optional, Sequence

try:
    from symspellpy import SymSpell, Verbosity
except ImportError:  # pragma: no cover - optional dependency
    SymSpell = None
    Verbosity = None

try:
    from rapidfuzz import fuzz as rf_fuzz
    from rapidfuzz import process as rf_process
except ImportError:  # pragma: no cover - optional dependency
    rf_fuzz = None
    rf_process = None


_SYMSPELL_CACHE: dict[tuple[tuple[str, ...], int, int], SymSpell] = {}


def _normalize_terms(terms: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for term in terms:
        if term is None:
            continue
        text = str(term).strip().lower()
        if text:
            normalized.append(text)
    return sorted(set(normalized))


def _expand_terms_with_tokens(terms: Iterable[str]) -> List[str]:
    expanded = set()
    for term in terms:
        if term is None:
            continue
        text = str(term).strip().lower()
        if not text:
            continue
        expanded.add(text)
        for token in re.findall(r"[a-z0-9]+", text):
            expanded.add(token)
    return sorted(expanded)


def _get_symspell(
    terms: Iterable[str],
    *,
    max_edit_distance: int,
    prefix_length: int,
) -> Optional[SymSpell]:
    if SymSpell is None:
        return None

    lexicon = tuple(_expand_terms_with_tokens(terms))
    cache_key = (lexicon, max_edit_distance, prefix_length)
    symspell = _SYMSPELL_CACHE.get(cache_key)
    if symspell is not None:
        return symspell

    symspell = SymSpell(
        max_dictionary_edit_distance=max_edit_distance,
        prefix_length=prefix_length,
    )
    for term in lexicon:
        symspell.create_dictionary_entry(term, 1)

    _SYMSPELL_CACHE[cache_key] = symspell
    return symspell


def normalize_question_variants(
    question: str,
    symspell_terms: Iterable[str],
    *,
    max_edit_distance: int = 2,
    prefix_length: int = 7,
) -> List[str]:
    if not question:
        return []

    question_lower = question.lower().strip()
    variants = [question_lower]

    symspell = _get_symspell(
        symspell_terms,
        max_edit_distance=max_edit_distance,
        prefix_length=prefix_length,
    )
    if symspell is None:
        return variants

    if Verbosity is None:
        return variants

    tokens = re.findall(r"[a-z0-9]+", question_lower)
    if not tokens:
        return variants

    corrected_tokens: List[str] = []
    for token in tokens:
        if token in symspell_terms or len(token) < 5 or token.isdigit():
            corrected_tokens.append(token)
            continue
        suggestions = symspell.lookup(token, Verbosity.CLOSEST, max_edit_distance)
        if suggestions:
            suggestion = suggestions[0]
            if suggestion.term and suggestion.term[0] == token[0]:
                corrected_tokens.append(suggestion.term)
            else:
                corrected_tokens.append(token)
        else:
            corrected_tokens.append(token)

    corrected = " ".join(corrected_tokens)
    if corrected and corrected != question_lower:
        ratio = difflib.SequenceMatcher(None, question_lower, corrected).ratio()
        if ratio >= 0.7:
            variants.append(corrected)

    return list(dict.fromkeys(variants))


def fuzzy_match_phrases(
    text: str,
    terms: Iterable[str],
    *,
    threshold: int = 90,
    max_ngram: int = 3,
    min_token_length: int = 3,
) -> List[str]:
    if not text:
        return []

    term_list = _normalize_terms(terms)
    if not term_list:
        return []

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return []

    matched: List[str] = []
    seen_terms: set[str] = set()
    seen_phrases: set[str] = set()

    for n in range(max_ngram, 0, -1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            if phrase in seen_phrases:
                continue
            seen_phrases.add(phrase)
            if len(phrase) < min_token_length:
                continue

            if rf_process and rf_fuzz:
                result = rf_process.extractOne(phrase, term_list, scorer=rf_fuzz.WRatio)
                match = result[0] if result and result[1] >= threshold else None
            else:
                matches = difflib.get_close_matches(
                    phrase,
                    term_list,
                    n=1,
                    cutoff=threshold / 100.0,
                )
                match = matches[0] if matches else None

            if match and match not in seen_terms:
                matched.append(match)
                seen_terms.add(match)

    return matched


__all__ = [
    "normalize_question_variants",
    "fuzzy_match_phrases",
]
