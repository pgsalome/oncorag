"""Helpers for filtering planned vs completed clinical evidence."""

from __future__ import annotations

from typing import Iterable
from pathlib import Path

import yaml


_DEFAULT_PLANNING_EVIDENCE_CUES = (
    "plan for",
    "plan to",
    "planned",
    "planning",
    "schedule",
    "scheduled",
    "will start",
    "will begin",
    "will receive",
    "will undergo",
    "to start",
    "to begin",
    "to receive",
    "to undergo",
    "candidate for",
    "consider",
    "considering",
    "considered",
    "recommend",
    "recommended",
    "referral",
    "consult",
    "consulted",
    "consultation",
    "discussed",
    "discussion",
    "pending",
    "awaiting",
)

_DEFAULT_COMPLETION_EVIDENCE_CUES = (
    "completed",
    "completion of",
    "received",
    "receiving",
    "underwent",
    "undergoing",
    "performed",
    "administered",
    "given",
    "started",
    "initiated",
    "finished",
    "ongoing",
    "in progress",
    "delivered",
    "s/p",
    "status post",
    "post-op",
    "postop",
    "post op",
)

_DEFAULT_NEGATIVE_EVIDENCE_CUES = (
    "declined",
    "refused",
    "did not",
    "didn't",
    "not receive",
    "no radiation",
    "no chemotherapy",
    "no chemo",
    "not given",
    "not administered",
    "never received",
    "cancelled",
    "canceled",
    "deferred",
    "withheld",
)

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "system_config.yaml"
if _CONFIG_PATH.exists():
    try:
        _CONFIG = yaml.safe_load(_CONFIG_PATH.read_text()) or {}
    except Exception:
        _CONFIG = {}
else:
    _CONFIG = {}


def _normalize_cues(values: Iterable[str]) -> tuple[str, ...]:
    cleaned = []
    for value in values or []:
        if value is None:
            continue
        text = str(value).strip().lower()
        if text:
            cleaned.append(text)
    return tuple(dict.fromkeys(cleaned))


_FILTERS = _CONFIG.get("evidence_filters") or {}
PLANNING_EVIDENCE_CUES = _normalize_cues(
    _FILTERS.get("planned") or _DEFAULT_PLANNING_EVIDENCE_CUES
)
COMPLETION_EVIDENCE_CUES = _normalize_cues(
    _FILTERS.get("completed") or _DEFAULT_COMPLETION_EVIDENCE_CUES
)
NEGATIVE_EVIDENCE_CUES = _normalize_cues(
    _FILTERS.get("negative") or _DEFAULT_NEGATIVE_EVIDENCE_CUES
)


def _contains_any(text: str, cues: Iterable[str]) -> bool:
    return any(cue in text for cue in cues)


def evidence_suggests_planning(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return _contains_any(lowered, PLANNING_EVIDENCE_CUES)


def evidence_suggests_completion(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return _contains_any(lowered, COMPLETION_EVIDENCE_CUES)


def evidence_suggests_negative(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return _contains_any(lowered, NEGATIVE_EVIDENCE_CUES)


def should_filter_planned_evidence(text: str) -> bool:
    """Return True when evidence looks planned without completion/negative cues."""
    if not text:
        return False
    if evidence_suggests_completion(text) or evidence_suggests_negative(text):
        return False
    return evidence_suggests_planning(text)


__all__ = [
    "should_filter_planned_evidence",
    "evidence_suggests_planning",
    "evidence_suggests_completion",
    "evidence_suggests_negative",
    "PLANNING_EVIDENCE_CUES",
    "COMPLETION_EVIDENCE_CUES",
    "NEGATIVE_EVIDENCE_CUES",
]
