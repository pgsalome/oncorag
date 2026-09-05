"""Run real local-model chat checks on the paired multilingual synthetic example cohorts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oncoraggraph.config.pipeline_config import load_pipeline_config, validate_pipeline_config
from oncoraggraph.ingestion import load_notes


VARIANTS = ("english", "german", "mixed")
TREATMENT_ALIASES = {
    "temozolomide": ("temozolomide", "temozolomid", "temodal", "temodar"),
    "bevacizumab": ("bevacizumab", "avastin"),
    "radiotherapy": ("radiotherapy", "radiation therapy", "strahlentherapie", "bestrahlung"),
}


def field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def response_payload(response: Any) -> dict[str, Any]:
    citations = []
    for item in field(response, "citations", []) or []:
        if is_dataclass(item):
            item = asdict(item)
        citations.append({key: field(item, key) for key in (
            "note_id", "quote", "date", "report_type", "language", "note_name", "note_date", "passage"
        ) if field(item, key) is not None})
    return {
        "answer": str(field(response, "answer", "") or ""),
        "reasoning": field(response, "reasoning", ""),
        "status": str(field(response, "status", "") or ""),
        "citations": citations,
    }


def verify_response(response: Any, notes: list, patient_id: str, treatment: str,
                    treatment_note_id: str, *, expected_date: str | None = None) -> list[str]:
    """Check answers and provenance independently against the authored notes."""
    failures = []
    answer = str(field(response, "answer", "") or "")
    status = str(field(response, "status", "") or "").lower()
    if not answer.strip() or status != "ok":
        failures.append("Response did not contain a supported answer")
    if expected_date is None:
        aliases = TREATMENT_ALIASES[treatment]
        if not any(re.search(r"\b" + re.escape(alias) + r"\b", answer, re.IGNORECASE) for alias in aliases):
            failures.append("Answer did not identify the expected started treatment")
    elif not re.search(r"(?<!\d)" + re.escape(expected_date) + r"(?!\d)", answer):
        failures.append("Follow-up did not give the correct treatment start date in ISO format")
    note_index = {note.note_id: note for note in notes if note.patient_id == patient_id}
    citations = field(response, "citations", []) or []
    valid_ids = set()
    if not citations:
        failures.append("Answer has no source-note citations")
    for citation in citations:
        note_id = field(citation, "note_id")
        note = note_index.get(note_id)
        if note is None:
            failures.append("Citation is not a note belonging to the selected patient")
            continue
        quote = str(field(citation, "quote", field(citation, "passage", "")) or "")
        if not quote.strip() or quote not in note.text:
            failures.append("Citation quote is not present in its source note")
            continue
        if field(citation, "date", field(citation, "note_date")) != note.date:
            failures.append("Citation date does not match authoritative registry metadata")
            continue
        if field(citation, "report_type") != note.report_type:
            failures.append("Citation report type does not match its source note")
            continue
        if field(citation, "language") != note.language:
            failures.append("Citation language does not match its source note")
            continue
        valid_ids.add(note_id)
    if treatment_note_id not in valid_ids:
        failures.append("The treatment-start report was not cited with valid evidence")
    return failures


def run_variant(config: dict, *, session_factory=None) -> dict[str, Any]:
    if session_factory is None:
        from oncoraggraph.chat_runtime import ChatSession
        session_factory = ChatSession
    inputs = config["inputs"]
    notes = load_notes(notes_root=inputs.get("notes_root"), registry_path=inputs.get("registry_path"))
    gold = [json.loads(line) for line in Path(config["evaluation"]["gold_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    treatments = {row["patient_id"]: row["value"] for row in gold if row["feature"] == "treatment_name"}
    patient_ids = ("SYN-DEMO-001", "SYN-DEMO-002")
    session = session_factory(config)
    failures = []
    turns = []
    language = config.get("cohort", {}).get("language", "en")
    questions = ("Welche Behandlung wurde tatsächlich begonnen?", "Wann wurde sie begonnen? Bitte YYYY-MM-DD verwenden.") \
        if language in {"de", "german"} else ("What treatment actually started?", "When did it start? Use YYYY-MM-DD.")
    for index, patient_id in enumerate(patient_ids):
        if patient_id not in session.patient_ids:
            failures.append(f"Synthetic example patient unavailable: {patient_id}")
            continue
        session.select_patient(patient_id)
        if session.patient_id != patient_id or session.history:
            failures.append("Selecting a patient did not reset conversation state")
        treatment_notes = [note for note in notes if note.patient_id == patient_id and note.report_type == "treatment"]
        if len(treatment_notes) != 1 or treatments.get(patient_id) not in TREATMENT_ALIASES:
            raise ValueError("Smoke input must use the authored single-treatment synthetic examples")
        treatment_note = treatment_notes[0]
        for position, question in enumerate(questions if index == 0 else questions[:1]):
            history_before = len(session.history)
            response = session.ask(question)
            turn_failures = verify_response(
                response, notes, patient_id, treatments[patient_id], treatment_note.note_id,
                expected_date=treatment_note.date if position else None,
            )
            if len(session.history) <= history_before:
                turn_failures.append("Chat did not retain the conversation turn")
            turns.append({"patient_id": patient_id, "question": question, "response": response_payload(response),
                          "checks_passed": not turn_failures, "failures": turn_failures})
            failures.extend(turn_failures)
    return {"passed": not failures, "turn_count": len(turns), "patients": list(patient_ids),
            "failures": failures, "turns": turns}


def run_smoke(ollama_host: str, ollama_model: str, output_dir: Path, *, session_factory=None) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant in VARIANTS:
        config = load_pipeline_config(ROOT / f"configs/oncorag_synthetic_{variant}.json")
        config["runtime"]["ollama"].update(host=ollama_host, model=ollama_model)
        config["outputs"]["root"] = str(output_dir / variant)
        config["features"]["generated_config_dir"] = str(output_dir / variant / "feature_configs")
        validate_pipeline_config(config)
        try:
            results[variant] = run_variant(config, session_factory=session_factory)
        except Exception as exc:
            results[variant] = {"passed": False, "turn_count": 0, "failures": [f"{type(exc).__name__}: {exc}"], "turns": []}
        (output_dir / f"{variant}.json").write_text(json.dumps(results[variant], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{variant}: {'passed' if results[variant]['passed'] else 'failed'}", flush=True)
    summary = {
        "model": ollama_model, "passed": all(result["passed"] for result in results.values()),
        "scope": "Nine real-model chat turns over paired synthetic example cohorts; not clinical performance validation",
        "variants": {variant: {key: result[key] for key in ("passed", "turn_count", "failures")} for variant, result in results.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="phi3:mini")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/chat_smoke")
    args = parser.parse_args(argv)
    summary = run_smoke(args.ollama_host, args.ollama_model, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
