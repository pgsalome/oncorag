"""Temporal data extraction for chatbot.

Extracts structured temporal data (value + date pairs) from the graph
for questions asking about measurements, treatments, or other time-series data.
This enables plotting and visualization of clinical data over time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from ..utils.logging_utils import log
from ..utils.evidence_utils import should_filter_planned_evidence, evidence_suggests_negative
from .term_matching import fuzzy_match_phrases, normalize_question_variants


@dataclass
class TemporalDataPoint:
    """A single temporal data point with value and date."""
    value: str | float | int
    date: Optional[str]
    unit: Optional[str] = None
    source: Optional[str] = None  # Node ID or note name
    context: Optional[str] = None  # Additional context


@dataclass
class TemporalDataSeries:
    """A series of temporal data points for a specific measurement/treatment."""
    name: str  # e.g., "BMI", "carboplatin dose"
    data_points: List[TemporalDataPoint]
    unit: Optional[str] = None


# Keywords that indicate temporal/quantitative queries
TEMPORAL_QUERY_INDICATORS = {
    "all", "over time", "timeline", "history", "dates", "when",
    "plot", "graph", "chart", "trend", "values", "measurements",
    "wann", "verlauf", "zeitverlauf", "werte", "messwerte", "diagramm", "datum",
}

QUANTITATIVE_MEASUREMENTS = {
    "bmi": ["bmi", "body mass index"],
    "hemoglobin": ["hemoglobin", "haemoglobin", "h\u00e4moglobin", "hb"],
    "weight": ["weight", "gewicht", "wt", "lbs", "kg", "pounds", "kilograms"],
    "height": ["height", "gr\u00f6\u00dfe", "groesse", "inches", "cm", "centimeters"],
    "blood pressure": [
        "blood pressure",
        "blutdruck",
        "blood presuure",
        "blood presure",
        "blood presssure",
        "bp",
        "systolic",
        "diastolic",
        "presuure",
        "presure",
        "presssure",
    ],
    "temperature": ["temperature", "temperatur", "temp", "fever"],
    "heart rate": ["heart rate", "herzfrequenz", "pulse", "puls", "bpm"],
    "dose": ["dose", "dosage", "mg", "mg/m2", "milligrams"],
    "dose per date": ["dose", "dosage", "mg", "mg/m2", "milligrams"],
}

TREATMENT_QUERY_INDICATORS = {
    "chemotherapy", "chemo", "drug", "medication", "treatment",
    "therapy", "regimen", "protocol",
}

CHEMOTHERAPY_DRUGS = [
    "carboplatin",
    "paclitaxel",
    "doxorubicin",
    "cyclophosphamide",
    "docetaxel",
    "taxol",
    "taxotere",
    "adriamycin",
    "cytoxan",
    "paraplatin",
    "abraxane",
    "carbo",
    "tax",
    "pembrolizumab",
    "keytruda",
    "pembro",
    "capecitabine",
    "xeloda",
]

CHEMO_DRUG_CANONICAL = {
    "carbo": "carboplatin",
    "paraplatin": "carboplatin",
    "tax": "paclitaxel",
    "taxol": "paclitaxel",
    "abraxane": "paclitaxel",
    "adriamycin": "doxorubicin",
    "cytoxan": "cyclophosphamide",
    "keytruda": "pembrolizumab",
    "pembro": "pembrolizumab",
    "xeloda": "capecitabine",
}

CHEMOTHERAPY_KEYWORDS = [
    "chemotherapy",
    "chemo",
    "antineoplastic",
    "regimen",
    "cycle",
    "infusion",
    "dose",
    "dosage",
]

TREATMENT_KEYWORDS = {
    "chemotherapy": CHEMOTHERAPY_DRUGS,
    "surgery": ["surgery", "surgical", "procedure", "operation", "mastectomy", "lumpectomy"],
    "radiation": ["radiation", "radiotherapy", "irradiation", "xrt", "imrt"],
}

SURGERY_EVENT_KEYWORDS = [
    ("mastectomy", ["mastectomy"]),
    ("lumpectomy", ["lumpectomy", "partial mastectomy"]),
    ("biopsy", ["biopsy", "biopsies"]),
    ("resection", ["resection", "excision"]),
    ("surgery", ["surgery", "surgical", "procedure", "operation", "operative"]),
]

RADIATION_EVENT_KEYWORDS = [
    ("external beam radiation", ["external beam", "ebrt"]),
    ("imrt", ["imrt", "intensity modulated"]),
    ("radiation therapy", ["radiation", "radiotherapy", "xrt", "irradiation"]),
]


def _build_measurement_keyword_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for measurement, keywords in QUANTITATIVE_MEASUREMENTS.items():
        mapping[measurement] = measurement
        for keyword in keywords:
            mapping[keyword] = measurement

    for keyword in ["chemotherapy", "chemo", "drug"]:
        mapping[keyword] = "chemotherapy"

    for keyword in TREATMENT_KEYWORDS.get("surgery", []):
        mapping[keyword] = "surgery"
    for keyword in ["surgery", "surgeries", "surgical", "procedure", "operation", "operative"]:
        mapping[keyword] = "surgery"

    for keyword in TREATMENT_KEYWORDS.get("radiation", []):
        mapping[keyword] = "radiation"
    for keyword in ["radiation", "radiotherapy", "irradiation"]:
        mapping[keyword] = "radiation"

    return mapping


_MEASUREMENT_KEYWORD_MAP = _build_measurement_keyword_map()
_TEMPORAL_SYMSPELL_TERMS = sorted(
    {
        *TEMPORAL_QUERY_INDICATORS,
        *_MEASUREMENT_KEYWORD_MAP.keys(),
    }
)


def is_temporal_query(question: str) -> bool:
    """Check if question is asking for temporal/quantitative data."""
    if not question:
        return False

    variants = normalize_question_variants(question, _TEMPORAL_SYMSPELL_TERMS)

    def _contains_indicator(indicator: str) -> bool:
        if " " in indicator:
            return any(indicator in variant for variant in variants)
        return any(
            re.search(rf"\b{re.escape(indicator)}\b", variant) is not None
            for variant in variants
        )

    # Check for temporal indicators
    if any(_contains_indicator(indicator) for indicator in TEMPORAL_QUERY_INDICATORS):
        return True

    # Check for quantitative measurements
    for measurement, keywords in QUANTITATIVE_MEASUREMENTS.items():
        if any(
            any(keyword in variant for keyword in keywords)
            for variant in variants
        ):
            if any(
                any(word in variant for word in ["all", "history", "over time", "dates"])
                for variant in variants
            ):
                return True

    # Check for treatment queries with temporal intent
    if any(
        any(treatment in variant for treatment in TREATMENT_QUERY_INDICATORS)
        for variant in variants
    ):
        if any(_contains_indicator(word) for word in ["all", "history", "dates", "when", "timeline"]):
            return True

    return False


def extract_measurement_name(question: str) -> Optional[str]:
    """Extract the measurement name from question."""
    if not question:
        return None

    variants = normalize_question_variants(question, _TEMPORAL_SYMSPELL_TERMS)

    # Check for specific measurements
    for measurement, keywords in QUANTITATIVE_MEASUREMENTS.items():
        if any(
            any(keyword in variant for keyword in keywords)
            for variant in variants
        ):
            return measurement

    # Check for treatment-related queries
    if any(
        any(treatment in variant for treatment in ["chemotherapy", "chemo", "drug"])
        for variant in variants
    ):
        return "chemotherapy"

    if any(
        any(term in variant for term in ["surgery", "surgeries", "surgical", "procedure"])
        for variant in variants
    ):
        return "surgery"

    if any(
        any(term in variant for term in ["radiation", "radiotherapy"])
        for variant in variants
    ):
        return "radiation"

    # Fuzzy match to measurement keywords
    for variant in variants:
        matches = fuzzy_match_phrases(variant, _MEASUREMENT_KEYWORD_MAP.keys())
        for match in matches:
            measurement = _MEASUREMENT_KEYWORD_MAP.get(match)
            if measurement:
                return measurement

    return None


def extract_temporal_data(
    graph: nx.Graph,
    nodes: List[str],
    measurement_name: Optional[str] = None,
) -> List[TemporalDataSeries]:
    """Extract temporal data (value + date pairs) from graph nodes.

    Args:
        graph: Patient knowledge graph
        nodes: List of relevant nodes to extract from
        measurement_name: Optional measurement name to filter by

    Returns:
        List of TemporalDataSeries with value + date pairs
    """
    data_series: Dict[str, List[TemporalDataPoint]] = {}

    chemo_mode = measurement_name and measurement_name.lower() in {
        "chemotherapy",
        "drug",
        "dose",
        "dose per date",
    }

    measurement_lower = measurement_name.lower() if measurement_name else None

    for node_id in nodes:
        if not graph.has_node(node_id):
            continue

        node_data = graph.nodes[node_id]

        if measurement_name and not _node_matches_measurement(node_data, node_id, measurement_name):
            continue

        text = _collect_node_text(node_data, node_id)
        if chemo_mode or measurement_lower in {"surgery", "radiation"}:
            if should_filter_planned_evidence(text) or evidence_suggests_negative(text):
                continue

        if chemo_mode:
            chemo_doses = _extract_chemo_doses(text)
            if not chemo_doses:
                continue
            date, source, context = _extract_date_source_context(node_id, node_data, graph)
            for drug, value, unit in chemo_doses:
                series_name = f"chemotherapy_{drug}"
                if series_name not in data_series:
                    data_series[series_name] = []
                data_series[series_name].append(
                    TemporalDataPoint(
                        value=value,
                        date=date,
                        unit=unit,
                        source=source,
                        context=context,
                    )
                )
            continue

        if measurement_lower in {"surgery", "radiation"}:
            event = _extract_treatment_event(text, measurement_lower)
            if not event:
                continue
            date, source, context = _extract_date_source_context(node_id, node_data, graph)
            series_name = measurement_lower
            if series_name not in data_series:
                data_series[series_name] = []
            data_series[series_name].append(
                TemporalDataPoint(
                    value=event,
                    date=date,
                    source=source,
                    context=context,
                )
            )
            continue

        # Extract value and date from node
        value, date, unit, source, context = _extract_value_date_from_node(
            node_id,
            node_data,
            graph,
            measurement_name=measurement_name,
        )

        if value is None:
            continue

        # Determine series name
        series_name = _determine_series_name(node_data, measurement_name)

        if series_name not in data_series:
            data_series[series_name] = []

        data_series[series_name].append(
            TemporalDataPoint(
                value=value,
                date=date,
                unit=unit,
                source=source,
                context=context,
            )
        )

    # Convert to TemporalDataSeries objects
    result: List[TemporalDataSeries] = []
    for name, points in data_series.items():
        # Sort by date if available
        sorted_points = _sort_by_date(points)
        result.append(
            TemporalDataSeries(
                name=name,
                data_points=sorted_points,
                unit=_extract_common_unit(points),
            )
        )

    log(
        f"Extracted {len(result)} temporal data series with {sum(len(s.data_points) for s in result)} data points",
        level="INFO",
        debug=True,
    )

    return result


def find_measurement_nodes(
    graph: nx.Graph,
    measurement_name: str,
    max_nodes: int = 400,
) -> List[str]:
    """Find nodes that mention a specific measurement or treatment."""
    matches: List[str] = []
    for node_id, node_data in graph.nodes(data=True):
        if _node_matches_measurement(node_data, node_id, measurement_name):
            matches.append(node_id)
            if len(matches) >= max_nodes:
                break
    return matches


def _extract_value_date_from_node(
    node_id: str,
    node_data: Dict,
    graph: nx.Graph,
    measurement_name: Optional[str] = None,
) -> Tuple[Optional[str | float | int], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract value, date, unit, source, and context from a node."""
    # Extract value
    value = None
    unit = None
    for key in ["value", "text", "sentence", "source_sentence", "original_text"]:
        val = node_data.get(key)
        if val:
            text_val = str(val)
            if measurement_name:
                parsed_value, parsed_unit = _extract_measurement(text_val, measurement_name)
                if parsed_value is not None:
                    value = parsed_value
                    unit = parsed_unit
                    break
                continue
            if isinstance(val, (int, float)):
                value = val
            else:
                # Check if it's a measurement-like string
                value = _parse_numeric_value(text_val)

    date, source, context = _extract_date_source_context(node_id, node_data, graph)

    return value, date, unit, source, context


def _parse_numeric_value(text: str) -> Optional[float | int]:
    """Parse numeric value from text."""
    if not text:
        return None

    # Remove common non-numeric prefixes/suffixes
    text = text.strip()

    # Try to extract number
    match = re.search(r'(?<![\d.,])[-+]?\d+(?:[.,]\d+)?(?!\d|[.,]\d)', text)
    if match:
        try:
            num_str = match.group().replace(",", ".")
            if '.' in num_str:
                return float(num_str)
            else:
                return int(num_str)
        except (ValueError, AttributeError):
            pass

    return None


def _determine_series_name(node_data: Dict, measurement_name: Optional[str]) -> str:
    """Determine the name of the data series."""
    if measurement_name:
        measurement_lower = measurement_name.lower()
        if measurement_lower in {"chemotherapy", "drug", "dose", "dose per date"}:
            text = _collect_node_text(node_data)
            drug = _extract_drug_name(text)
            if drug:
                return f"chemotherapy_{drug}"
            return "chemotherapy"
        return measurement_lower

    # Try to infer from node label
    label = node_data.get("label", "")
    if label:
        return label.lower()

    # Try to infer from text
    text = str(node_data.get("text", "") or node_data.get("value", "") or "")
    text_lower = text.lower()

    # Check for common measurements
    for measurement, keywords in QUANTITATIVE_MEASUREMENTS.items():
        if any(keyword in text_lower for keyword in keywords):
            return measurement

    # Check for treatments
    if any(treatment in text_lower for treatment in ["chemotherapy", "chemo"]):
        # Try to extract drug name
        drug_match = re.search(
            r'\b(carboplatin|paclitaxel|doxorubicin|cyclophosphamide|docetaxel|taxol|taxotere|adriamycin|cytoxan|paraplatin|abraxane)\b',
            text_lower,
        )
        if drug_match:
            return f"chemotherapy_{drug_match.group(1)}"
        return "chemotherapy"

    if "surgery" in text_lower or "mastectomy" in text_lower or "lumpectomy" in text_lower:
        return "surgery"

    if "radiation" in text_lower or "radiotherapy" in text_lower:
        return "radiation"

    return "measurement"


def _collect_node_text(node_data: Dict, node_id: Optional[str] = None) -> str:
    parts: List[str] = []
    for key in ("label", "type", "text", "sentence", "source_sentence", "value", "description", "original_text"):
        value = node_data.get(key)
        if value:
            parts.append(str(value))
    if node_id:
        parts.append(str(node_id))
    return " ".join(parts).lower()


def _extract_date_source_context(
    node_id: str,
    node_data: Dict,
    graph: nx.Graph,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Extract date from node or neighbors
    date = None
    date_node = None

    for key in ["date", "note_date", "timestamp"]:
        date_val = node_data.get(key)
        if date_val:
            date = str(date_val)
            break

    if not date:
        for neighbor in graph.neighbors(node_id):
            if not graph.has_node(neighbor):
                continue
            neighbor_data = graph.nodes[neighbor]
            neighbor_label = neighbor_data.get("label", "")
            if neighbor_label == "Date":
                date_val = neighbor_data.get("value") or neighbor_data.get("text") or str(neighbor)
                if date_val:
                    date = str(date_val)
                    date_node = neighbor
                    break

    source = node_data.get("note_name") or node_data.get("note_id") or None

    context = node_data.get("sentence") or node_data.get("text") or node_data.get("source_sentence")
    if context and len(str(context)) > 200:
        context = str(context)[:197] + "..."

    if date and context:
        context_lower = str(context).lower()
        if "dob" in context_lower or "date of birth" in context_lower or "born" in context_lower:
            date = None

    return date, source, context


def _extract_treatment_event(text: str, measurement_lower: str) -> Optional[str]:
    if not text:
        return None
    text_lower = text.lower()
    if measurement_lower == "surgery":
        for label, keywords in SURGERY_EVENT_KEYWORDS:
            if any(keyword in text_lower for keyword in keywords):
                return label
    if measurement_lower == "radiation":
        for label, keywords in RADIATION_EVENT_KEYWORDS:
            if any(keyword in text_lower for keyword in keywords):
                return label
    return None


def _normalize_chemo_drug(drug: str) -> str:
    return CHEMO_DRUG_CANONICAL.get(drug, drug)


def _extract_chemo_doses(text: str) -> List[Tuple[str, float | int, Optional[str]]]:
    text_lower = text.lower()
    results: List[Tuple[str, float | int, Optional[str]]] = []
    for drug in CHEMOTHERAPY_DRUGS:
        pattern = rf'\b{re.escape(drug)}\b[^0-9]{{0,24}}(\d+(?:[.,]\d+)?)\s*(mg/m\^?2|mg|g|mcg|auc)\b'
        for match in re.finditer(pattern, text_lower):
            value = _parse_numeric_value(match.group(1))
            unit = match.group(2).replace("mg/m^2", "mg/m2")
            if value is None:
                continue
            results.append((_normalize_chemo_drug(drug), value, unit))
    # De-duplicate while preserving order
    seen = set()
    unique_results = []
    for drug, value, unit in results:
        key = (drug, value, unit)
        if key in seen:
            continue
        seen.add(key)
        unique_results.append((drug, value, unit))
    return unique_results


def _node_matches_measurement(node_data: Dict, node_id: str, measurement_name: str) -> bool:
    measurement_lower = measurement_name.lower()
    text = _collect_node_text(node_data, node_id)

    if measurement_lower in QUANTITATIVE_MEASUREMENTS:
        keywords = QUANTITATIVE_MEASUREMENTS[measurement_lower]
        return any(keyword in text for keyword in keywords)

    if measurement_lower in {"chemotherapy", "drug", "dose", "dose per date"}:
        return any(keyword in text for keyword in CHEMOTHERAPY_DRUGS)

    if measurement_lower in TREATMENT_KEYWORDS:
        return any(keyword in text for keyword in TREATMENT_KEYWORDS[measurement_lower])

    return measurement_lower in text


def _extract_drug_name(text: str) -> Optional[str]:
    text_lower = text.lower()
    for drug in CHEMOTHERAPY_DRUGS:
        if drug in text_lower:
            return drug
    return None


def _extract_measurement_value(text: str, measurement_name: Optional[str]) -> Optional[str | float | int]:
    return _extract_measurement(text, measurement_name)[0]


_NUMBER = r"(?<![\w.,])[-+]?\d+(?:[.,]\d+)?(?!\d|[.,]\d)"
_LABEL_SEPARATOR = r"\s*(?:(?:measured\s+today|heute|today|is|was|of|at|betr\u00e4gt|betrug|von)\s*)?[:=]?\s*"
_UNIT_END = r"(?![\w/^])"


def _extract_measurement(text: str, measurement_name: Optional[str]):
    """Return a value and its own unit, never a neighboring measurement's unit."""
    if not text:
        return None, None

    if not measurement_name:
        return _parse_numeric_value(text), None

    measurement_lower = measurement_name.lower()
    text_lower = text.lower()

    if measurement_lower == "blood pressure":
        pressure = r'(\d{2,3})\s*/\s*(\d{2,3})\b'
        match = re.search(rf'\b(?:blood pressure|blutdruck|bp)\b{_LABEL_SEPARATOR}{pressure}', text_lower)
        if match is None:
            match = re.search(rf'\b{pressure}\s*mmhg\b', text_lower)
        if match:
            systolic = int(match.group(1))
            diastolic = int(match.group(2))
            if 50 <= systolic <= 250 and 30 <= diastolic <= 150:
                unit = "mmHg" if "mmhg" in match.group() or re.match(r"\s*mmhg\b", text_lower[match.end():]) else None
                return f"{systolic}/{diastolic}", unit
        return None, None

    definitions = {
        "hemoglobin": (r"hemoglobin|haemoglobin|h\u00e4moglobin|hb", r"g/dl|g/l|mmol/l", False),
        "bmi": (r"bmi|body mass index", r"kg/m\^?2", False),
        "weight": (r"weight|body weight|gewicht|k\u00f6rpergewicht|wt", r"kilograms?|kg|pounds?|lbs?|lb", True),
        "height": (r"height|body height|gr\u00f6\u00dfe|groesse|k\u00f6rpergr\u00f6\u00dfe|ht",
                   r"centimeters?|centimetres?|cm|meters?|metres?|m|inches|inch|in|feet|ft", True),
        "temperature": (r"temperature|temperatur|temp", r"\u00b0\s*c|\u00b0\s*f", True),
        "heart rate": (r"heart rate|herzfrequenz|pulse|puls|hr", r"bpm|/min", True),
    }
    if measurement_lower in {"chemotherapy", "drug", "dose", "dose per date"}:
        definitions[measurement_lower] = (r"dose|dosage|dosis", r"mg/m\^?2|mcg|mg|auc|g", True)
    if measurement_lower not in definitions:
        return None, None
    labels, units, unit_required = definitions[measurement_lower]
    unit_group = rf"\s*(?P<unit>{units}){_UNIT_END}"
    if not unit_required:
        unit_group = rf"(?:{unit_group})?"
    pattern = rf"\b(?:{labels})\b{_LABEL_SEPARATOR}(?P<value>{_NUMBER}){unit_group}"
    match = re.search(pattern, text_lower)
    if match is None and measurement_lower in {"weight", "height"}:
        # Units such as kg/cm carry a metric identity; reject kg/m2 and ambiguous "in".
        fallback_units = units.replace("|in|", "|") if measurement_lower == "height" else units
        match = re.search(rf"(?P<value>{_NUMBER})\s*(?P<unit>{fallback_units}){_UNIT_END}", text_lower)
    if match is None:
        return None, None
    unit = match.group("unit")
    normalized = {"g/dl": "g/dL", "g/l": "g/L", "mmol/l": "mmol/L", "mg/m^2": "mg/m2",
                  "kg/m^2": "kg/m2", "\u00b0c": "\u00b0C", "\u00b0f": "\u00b0F"}
    unit = normalized.get(unit.replace(" ", ""), unit) if unit else None
    if measurement_lower == "bmi":
        unit = "kg/m2"
    return _parse_numeric_value(match.group("value")), unit


def _sort_by_date(points: List[TemporalDataPoint]) -> List[TemporalDataPoint]:
    """Sort data points by date."""
    def date_key(point: TemporalDataPoint) -> Tuple[int, int, int]:
        if not point.date:
            return (9999, 99, 99)  # Put dates without date at end

        # Try to parse date (simple YYYY-MM-DD or YYYY/MM/DD format)
        date_str = str(point.date)
        match = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', date_str)
        if match:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return (year, month, day)

        # Try YYYY format
        match = re.search(r'(\d{4})', date_str)
        if match:
            return (int(match.group(1)), 1, 1)

        return (9999, 99, 99)

    return sorted(points, key=date_key)


def _extract_common_unit(points: List[TemporalDataPoint]) -> Optional[str]:
    """Extract the most common unit from data points."""
    units = [p.unit for p in points if p.unit]
    if not units:
        return None

    # Return most common unit
    from collections import Counter
    unit_counts = Counter(units)
    return unit_counts.most_common(1)[0][0]


def format_temporal_data_for_json(series: List[TemporalDataSeries]) -> Dict:
    """Format temporal data as JSON-serializable dict for plotting."""
    result = {
        "series": []
    }

    for s in series:
        series_data = {
            "name": s.name,
            "unit": s.unit,
            "data": []
        }

        for point in s.data_points:
            point_data = {
                "value": point.value,
                "date": point.date,
            }
            if point.unit:
                point_data["unit"] = point.unit
            if point.source:
                point_data["source"] = point.source
            if point.context:
                point_data["context"] = point.context

            series_data["data"].append(point_data)

        result["series"].append(series_data)

    return result


__all__ = [
    "is_temporal_query",
    "extract_measurement_name",
    "extract_temporal_data",
    "find_measurement_nodes",
    "TemporalDataSeries",
    "TemporalDataPoint",
    "format_temporal_data_for_json",
]
