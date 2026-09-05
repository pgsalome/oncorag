"""Patient-scoped chat sessions using the portable extraction configuration."""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from .config.pipeline_config import load_pipeline_config, validate_pipeline_config
from .pipeline import (
    prepare_features,
    prepare_inputs,
    prepare_patient_graph,
    prepare_patient_index,
    seed_runtime,
)


class ChatSession:
    """Keep graphs and conversation state local to one caller and one patient."""

    def __init__(self, config, *, graph_builder=None, collection_factory=None,
                 indexer=None, service_factory=None):
        validate_pipeline_config(config)
        self.config = copy.deepcopy(config)
        self._specs, self._patients = prepare_inputs(self.config)
        self._graph_builder = graph_builder
        self._collection_factory = collection_factory
        self._indexer = indexer
        self._service_factory = service_factory
        self.patient_id = None
        self.graph = None
        self.collection = None
        self.graph_path = None
        self.graph_fingerprint = None
        self._service = None
        self._history = []

    @property
    def patient_ids(self):
        return sorted(self._patients)

    @property
    def history(self):
        return copy.deepcopy(self._history)

    def reset(self):
        """Forget conversation history without discarding the current graph."""
        self._history.clear()

    def close(self):
        """Drop all references to the previously selected patient."""
        self.reset()
        self.patient_id = None
        self.graph = None
        self.collection = None
        self.graph_path = None
        self.graph_fingerprint = None
        self._service = None

    def select_patient(self, patient_id, *, force_rebuild=False):
        # Invalidate before validation or I/O so a failed switch cannot expose old data.
        self.close()
        self._specs, self._patients = prepare_inputs(self.config)
        if not isinstance(patient_id, str) or patient_id not in self._patients:
            raise ValueError("Selected patient is not present in the configured inputs")
        prepare_features(self.config, self._specs)
        seed_runtime(self.config["runtime"])
        graph, graph_path, content_key = prepare_patient_graph(
            self.config, patient_id, self._patients[patient_id],
            graph_builder=self._graph_builder, force_rebuild=force_rebuild,
        )
        collection = prepare_patient_index(
            self.config, patient_id, graph,
            collection_factory=self._collection_factory, indexer=self._indexer,
        )
        factory = self._service_factory
        if factory is None:
            from .chat import ChatGraphService
            factory = ChatGraphService
        service = factory(
            Path(self.config["features"]["generated_config_dir"]),
            runtime_config=self.config["runtime"], retrieval_config=self.config["retrieval"],
            chat_config=self.config.get("chat", {}),
        )
        self.patient_id = patient_id
        self.graph, self.graph_path, self.graph_fingerprint = graph, graph_path, content_key
        self.collection, self._service = collection, service

    def ask(self, question):
        if self.patient_id is None or self._service is None:
            raise ValueError("Select a patient before asking a question")
        settings = self.config.get("chat", {})
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a nonempty string")
        if len(question) > settings.get("max_question_chars", 4000):
            raise ValueError("Question exceeds chat.max_question_chars")
        if self.graph.graph.get("patient_id") != self.patient_id:
            self.close()
            raise ValueError("Selected graph no longer matches the active patient")
        response = self._service.answer_question(
            self.patient_id, self.graph, self.collection, question.strip(), history=self.history,
        )
        if response.status in {"ok", "missing"}:
            self._history.extend([
                {"role": "user", "content": question.strip()},
                {"role": "assistant", "content": response.answer},
            ])
            maximum = settings.get("history_turns", 5) * 2
            self._history = self._history[-maximum:] if maximum else []
            while self._history and sum(len(row["content"]) for row in self._history) > settings.get("max_history_chars", 12000):
                del self._history[:2]
        return response


def _show_response(response, json_output=False):
    if json_output:
        print(json.dumps(asdict(response), ensure_ascii=False, indent=2, allow_nan=False))
        return
    print(f"\nOncoRAG: {response.answer}")
    if response.status not in {"ok", "missing"}:
        print(f"Status: {response.status}")
        print(response.reasoning or "No answer was produced.")
    for citation in response.citations:
        note_id = citation.get("note_id", citation.get("note_name", ""))
        date = citation.get("date", citation.get("note_date", ""))
        text = citation.get("quote", citation.get("passage", ""))
        print(f"[{note_id} | {date}] {text}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patient", nargs="?", help="Exact patient ID (or use --patient-id)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--patient-id")
    parser.add_argument("--list-patients", action="store_true")
    parser.add_argument("-q", "--question")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print structured single-question output")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--ollama-host")
    parser.add_argument("--ollama-model")
    parser.add_argument("--vector-backend", choices=["chroma", "iris"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.patient and args.patient_id and args.patient != args.patient_id:
        parser.error("Specify only one patient ID")
    patient_id = args.patient_id or args.patient
    if not args.list_patients and not patient_id:
        parser.error("--patient-id is required unless --list-patients is used")
    if args.json and (not args.question or args.loop):
        parser.error("--json requires one --question and cannot be used with --loop")
    try:
        config = load_pipeline_config(args.config)
        for key, env in (("host", "OLLAMA_HOST"), ("model", "OLLAMA_MODEL")):
            value = getattr(args, "ollama_" + key) or os.environ.get(env)
            if value:
                config["runtime"]["ollama"][key] = value
        if args.vector_backend:
            config.setdefault("vector_store", {})["backend"] = args.vector_backend
        from .utils.logging_utils import set_quiet_mode
        set_quiet_mode(not args.verbose)
        session = ChatSession(config)
        if args.list_patients:
            print("\n".join(session.patient_ids))
            return 0
        try:
            with redirect_stdout(sys.stderr) if args.json else nullcontext():
                session.select_patient(patient_id, force_rebuild=args.force_rebuild)
            failures = False
            if args.question:
                with redirect_stdout(sys.stderr) if args.json else nullcontext():
                    response = session.ask(args.question)
                _show_response(response, args.json)
                failures = response.status in {"error", "invalid"}
            if not args.question or args.loop:
                print("Enter a question; /clear resets history; /quit exits.")
                while True:
                    try:
                        question = input("you> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if not question or question == "/quit":
                        break
                    if question == "/clear":
                        session.reset()
                        print("Conversation cleared.")
                        continue
                    response = session.ask(question)
                    _show_response(response)
                    failures |= response.status in {"error", "invalid"}
            return int(failures)
        finally:
            session.close()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Chat failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
