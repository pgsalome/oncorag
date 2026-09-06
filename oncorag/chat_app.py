"""Optional, session-scoped Streamlit interface for patient graph questions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from oncorag.config.pipeline_config import load_pipeline_config, validate_pipeline_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "oncorag_synthetic_mixed.json"


def _create_session(config):
    from oncorag.chat_runtime import ChatSession

    return ChatSession(config)


def _config_signature(path):
    try:
        source = Path(path).expanduser().resolve()
        return str(source), hashlib.sha256(source.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        return str(path), type(exc).__name__


@dataclass
class ChatUIState:
    signature: object = None
    config: dict | None = None
    session: object = None
    requested_patient: str | None = None
    transcript: list = field(default_factory=list)
    error: str | None = None

    def _discard_session(self):
        previous, self.session = self.session, None
        self.requested_patient = None
        self.transcript.clear()
        if previous is not None:
            for method in ("reset", "close"):
                callback = getattr(previous, method, None)
                if callable(callback):
                    try:
                        callback()
                    except Exception:
                        pass

    def configure(self, path, signature, *, ollama_host=None, ollama_model=None):
        self._discard_session()
        self.config, self.error = None, None
        self.signature = signature
        try:
            self.config = load_pipeline_config(Path(path).expanduser())
            if ollama_host is not None:
                self.config["runtime"]["ollama"]["host"] = ollama_host
            if ollama_model is not None:
                self.config["runtime"]["ollama"]["model"] = ollama_model
            validate_pipeline_config(self.config)
            self.session = _create_session(self.config)
        except Exception as exc:
            self.config, self.session = None, None
            self.error = f"Configuration could not be loaded: {exc}"

    def select_patient(self, patient_id, *, force_rebuild=False):
        self.transcript.clear()
        self.error = None
        self.requested_patient = patient_id
        try:
            if patient_id is None:
                self._discard_session()
                self.session = _create_session(self.config)
            else:
                self.session.select_patient(patient_id, force_rebuild=force_rebuild)
        except Exception as exc:
            self.error = f"Patient could not be loaded: {exc}"

    @property
    def ready(self):
        return (
            self.session is not None and self.error is None
            and self.requested_patient is not None
            and self.session.patient_id == self.requested_patient
        )

    def reset_conversation(self):
        self.transcript.clear()
        self.error = None
        if self.session is not None:
            self.session.reset()


def _temporal_rows(temporal_data):
    rows = []
    if not isinstance(temporal_data, dict):
        return rows
    series_list = temporal_data.get("series", [])
    if not isinstance(series_list, list):
        return rows
    for series in series_list:
        if not isinstance(series, dict):
            continue
        points = series.get("data", [])
        if not isinstance(points, list):
            continue
        for point in points:
            if not isinstance(point, dict):
                continue
            row = {"series": str(series.get("name") or "Measurement"),
                   "date": point.get("date"), "value": point.get("value"),
                   "unit": point.get("unit") or series.get("unit") or ""}
            for key in ("source", "context"):
                if point.get(key):
                    row[key] = point[key]
            rows.append(row)
    return rows


def _render_response(response):
    status = getattr(response, "status", "ok")
    if status in {"error", "invalid"}:
        st.error(response.reasoning or response.answer or "No answer was produced.")
        return
    elif status == "missing":
        st.warning(response.answer)
    else:
        st.markdown(response.answer)
    if response.reasoning:
        with st.expander("Reasoning"):
            st.write(response.reasoning)
    if response.citations:
        with st.expander("Evidence", expanded=True):
            for citation in response.citations:
                source = citation.get("note_name") or citation.get("note_id") or citation.get("node") or "Report"
                metadata = [str(source), str(citation.get("note_date") or citation.get("date") or "Date unknown")]
                if citation.get("report_type"):
                    metadata.append(str(citation["report_type"]))
                if citation.get("language"):
                    metadata.append(str(citation["language"]))
                st.caption(" | ".join(metadata))
                st.text(citation.get("passage") or citation.get("quote") or "")
    rows = _temporal_rows(response.temporal_data)
    if rows:
        with st.expander("Timeline", expanded=True):
            frame = pd.DataFrame(rows)
            for (name, unit), series in frame.groupby(["series", "unit"], sort=False):
                values = series.copy()
                values["date"] = pd.to_datetime(values["date"], errors="coerce")
                values["value"] = pd.to_numeric(values["value"], errors="coerce")
                values = values.dropna(subset=["date", "value"]).sort_values("date")
                if not values.empty:
                    st.caption(f"{name} ({unit})" if unit else name)
                    st.line_chart(values, x="date", y="value")
            st.dataframe(frame, hide_index=True, width="stretch")
    if response.medical_definitions:
        with st.expander("Medical definitions"):
            st.json(response.medical_definitions)
    if response.ontology_citations:
        with st.expander("Ontology sources"):
            st.json(response.ontology_citations)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--ollama-host")
    parser.add_argument("--ollama-model")
    args, _ = parser.parse_known_args(argv)
    ollama_host = args.ollama_host or os.environ.get("OLLAMA_HOST") or None
    ollama_model = args.ollama_model or os.environ.get("OLLAMA_MODEL") or None
    st.set_page_config(page_title="OncoRAG Chat", page_icon=":material/forum:", layout="wide")
    if "oncorag_chat_state" not in st.session_state:
        st.session_state.oncorag_chat_state = ChatUIState()
    state = st.session_state.oncorag_chat_state

    with st.sidebar:
        st.header("Patient Workspace")
        config_path = st.text_input("Parameter config", value=args.config, key="oncorag_chat_config")
        reload_config = st.button("Reload configuration", icon=":material/settings_backup_restore:")
        signature = (_config_signature(config_path), ollama_host, ollama_model)
        if state.signature != signature or reload_config:
            state.configure(config_path, signature, ollama_host=ollama_host, ollama_model=ollama_model)
            st.session_state.pop("oncorag_chat_patient", None)
        if state.session is not None:
            patient_id = st.selectbox("Patient", options=state.session.patient_ids, index=None,
                                      placeholder="Select patient", key="oncorag_chat_patient")
            reload_patient = st.button("Reload patient", icon=":material/refresh:", disabled=patient_id is None)
            if patient_id != state.requested_patient or reload_patient:
                with st.spinner("Loading patient..."):
                    state.select_patient(patient_id, force_rebuild=reload_patient)
        if st.button("Clear conversation", icon=":material/delete_sweep:", disabled=not state.transcript):
            state.reset_conversation()
            st.rerun()

    st.title("OncoRAG Chat")
    st.warning("Research use only. Responses may be incorrect; verify against the original reports.", icon=":material/science:")
    if state.ready:
        st.caption(f"Patient: {state.session.patient_id}")
    if state.error:
        st.error(state.error)
    for message in state.transcript:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            elif "error" in message:
                st.error(message["error"])
            else:
                _render_response(message["content"])

    question = st.chat_input("Question about this patient's reports", disabled=not state.ready)
    if question and state.ready:
        state.transcript.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Reviewing evidence..."):
                    response = state.session.ask(question)
                state.transcript.append({"role": "assistant", "content": response})
            except Exception as exc:
                state.transcript.append({"role": "assistant", "error": f"The question could not be answered: {exc}"})
        st.rerun()


if __name__ == "__main__":
    main()
