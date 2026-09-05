#!/usr/bin/env python3
"""Prepare a clean, allowlisted source snapshot without copying Git history."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml


RUNTIME_FILES = (
    "__init__.py", "main.py", "pipeline.py", "create_config.py", "ingestion.py", "evaluation.py",
    "config/__init__.py", "config/system_config.py", "config/pipeline_config.py", "config/feature_schema.py",
    "graph/__init__.py", "graph/graph_builder.py", "graph/graph_utils.py",
    "models/__init__.py", "models/model_init.py", "models/entity_extraction.py",
    "llm/__init__.py", "llm/llm_adapter.py", "llm/llm_query.py", "llm/prompt_builder.py",
    "retrieval/__init__.py", "retrieval/multi_stage.py", "rerank/graphrag_reranker.py",
    "chroma/__init__.py", "chroma/chroma_index.py",
    "vector_store/__init__.py", "vector_store/config.py", "vector_store/backend.py",
    "vector_store/iris.py", "vector_store/records.py",
    "utils/__init__.py", "utils/logging_utils.py", "utils/file_utils.py", "utils/parsing_utils.py",
    "utils/evidence_utils.py", "utils/phi_removal.py", "utils/scispacy_entities.py",
)
ROOT_FILES = ("README.md", "pyproject.toml", "setup.py", "LICENSE", "LICENSE.txt", "CITATION.cff", ".gitignore")
# Binary assets must remain byte-identical to the individually reviewed originals.
REVIEWED_ASSETS = {
    "graphicalabstract.png": "b23335b44752ff857becc554f683053e6b9d7fd031eeb78b2ba0bb008bbe5864",
}
CHAT_FILES = (
    "chat_runtime.py", "chat_app.py", "chat/__init__.py", "chat/service.py",
    "chat/medical_definitions.py", "chat/reranking.py", "chat/context_extraction.py",
    "chat/retrieval.py", "chat/query_expansion.py", "chat/term_matching.py",
    "chat/temporal_extraction.py",
)
SCRIPT_FILES = (
    "run_oncorag_full_pipeline.py", "export_synthetic_datasets.py", "prepare_public_release.py",
    "evaluate_synthetic.py",
    "run_synthetic_smoke.py",
    "run_chat_smoke.py",
)
PRIVATE_FIELDS = {
    "source_style_patient_id", "source_style_patient_ids", "style_source_patient_ids",
    "sampled_source_patient_ids", "style_patient_ids", "style_source_root",
}
SECRET_FIELDS = {"password", "api_key", "access_token", "client_secret", "secret_key", "private_key"}
SECRET_MARKERS = re.compile(
    r"\bsk-[A-Za-z0-9_-]{20,}|\bgh[pousr]_[A-Za-z0-9]{20,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
HOME_PATH = re.compile(r"/(?:home|Users)/[^\s\"'<>]+|[A-Za-z]:\\Users\\[^\s\"'<>]+")
DATASET_EXTENSIONS = {".txt", ".csv", ".json", ".jsonl", ".md"}


def validate_payload(value: Any) -> None:
    """Inspect structured metadata recursively, without echoing rejected values."""
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key).lower()
            if name in PRIVATE_FIELDS:
                raise ValueError("Private source metadata field in release input")
            if name in SECRET_FIELDS and item not in (None, ""):
                if not isinstance(item, str) or not re.fullmatch(r"\$\{[A-Z0-9_]+\}", item):
                    raise ValueError("Populated credential field in release input")
            validate_payload(item)
    elif isinstance(value, list):
        for item in value:
            validate_payload(item)


def validate_file(relative: str, contents: bytes) -> None:
    if relative in REVIEWED_ASSETS:
        if hashlib.sha256(contents).hexdigest() != REVIEWED_ASSETS[relative]:
            raise ValueError("Reviewed asset does not match its approved SHA-256")
        return
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Unexpected non-text release input") from exc
    if SECRET_MARKERS.search(text):
        raise ValueError("Credential marker in release input")
    for match in HOME_PATH.finditer(text):
        # The source-ID rejection test intentionally uses a fictitious home path.
        if relative.startswith("tests/") and match.group().startswith("/" + "home/person/"):
            continue
        raise ValueError("Absolute user-home path in release input")
    suffix = Path(relative).suffix
    if suffix == ".json":
        validate_payload(json.loads(text))
    elif suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                validate_payload(json.loads(line))
    elif suffix in {".yaml", ".yml"}:
        validate_payload(yaml.safe_load(text))
    elif suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        validate_payload({name: None for name in reader.fieldnames or []})
        for row in reader:
            validate_payload(row)


def _source_file(root: Path, relative: str) -> Path:
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Release source symlink escapes the repository") from exc
    return path


def select_files(source: Path, include_datasets: bool = False) -> dict[str, Path]:
    selected: dict[str, Path] = {}

    def include(relative: str, *, target: str | None = None, required: bool = False) -> None:
        path = _source_file(source, relative)
        if path.is_file():
            selected[target or relative] = path
        elif required:
            raise FileNotFoundError(f"Required release file is missing: {relative}")

    for name in ROOT_FILES:
        include(name, required=name in {"README.md", "pyproject.toml"})
    for name in REVIEWED_ASSETS:
        include(name)
    for name in RUNTIME_FILES:
        include("oncoraggraph/" + name, required=True)
    if (source / "oncoraggraph/chat_runtime.py").is_file():
        for name in CHAT_FILES:
            include("oncoraggraph/" + name)
        wrapper = source / "run_chatbot.py"
        if wrapper.is_file():
            wrapper_tree = ast.parse(wrapper.read_text(encoding="utf-8"))
            if not any(isinstance(node, ast.ImportFrom) and node.module == "oncoraggraph.chat_runtime"
                       for node in ast.walk(wrapper_tree)):
                raise ValueError("Public run_chatbot.py must import the portable chat runtime")
            include("run_chatbot.py")
        include("streamlit_app.py")
    public_system = "configs/system.public.yaml"
    if (source / public_system).is_file():
        include(public_system)
    else:
        public_system = "oncoraggraph/system_config.yaml"
    include(public_system, target="oncoraggraph/system_config.yaml", required=True)
    for name in SCRIPT_FILES:
        include("scripts/" + name)
    for path in sorted((source / "tests").glob("test_*.py")):
        include(path.relative_to(source).as_posix())
    include("tests/conftest.py")
    include(".github/workflows/tests.yml")
    include("examples/features.synthetic.yaml", required=True)
    include("examples/datasets/README.md")
    for pattern in ("oncorag*.example.json", "oncorag_synthetic_*.json", "synthetic*.json", "vector_store.iris.example.yaml"):
        for path in sorted((source / "configs").glob(pattern)):
            include(path.relative_to(source).as_posix())
    dataset_roots = ["examples/datasets/fixtures"]
    if include_datasets:
        dataset_roots += ["examples/datasets/english", "examples/datasets/german"]
    for relative in dataset_roots:
        root = _source_file(source, relative)
        if not root.is_dir():
            raise FileNotFoundError(f"Required dataset directory is missing: {relative}")
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in DATASET_EXTENSIONS:
                include(path.relative_to(source).as_posix())
    return selected


def validate_local_imports(source: Path, selected: dict[str, Path]) -> None:
    """Catch existing local Python modules missing from the explicit allowlist."""
    for relative, path in selected.items():
        if not relative.startswith("oncoraggraph/") or path.suffix != ".py":
            continue
        module = relative[:-3].replace("/", ".")
        package = module.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            candidates = []
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split(".")
                    base = ".".join(parts[: len(parts) - node.level + 1])
                    base += ("." + node.module) if node.module else ""
                else:
                    base = node.module or ""
                candidates = [base] + [base + "." + alias.name for alias in node.names]
            for name in candidates:
                if not name.startswith("oncoraggraph"):
                    continue
                for candidate in (name.replace(".", "/") + ".py", name.replace(".", "/") + "/__init__.py"):
                    if (source / candidate).is_file() and candidate not in selected:
                        raise ValueError(f"Runtime import missing from release allowlist: {candidate}")


def read_release_contents(selected: dict[str, Path]) -> dict[str, bytes]:
    contents = {}
    for relative, path in selected.items():
        data = path.read_bytes()
        try:
            validate_file(relative, data)
        except (ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"Release screening failed for {relative}: {exc}") from exc
        contents[relative] = data
    return contents


def build_manifest(contents: dict[str, bytes], include_datasets: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "Local review snapshot; no Git history, commit or publication operation",
        "full_synthetic_datasets_included": include_datasets,
        "publication_status": "Review code/data licensing and provenance before publication",
        "screening_scope": "Explicit file allowlist, SHA-256-pinned reviewed assets, local import closure, credential/home-path markers, selected private metadata keys",
        "file_count": len(contents),
        "payload_bytes": sum(len(data) for data in contents.values()),
        "files": {
            relative: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for relative, data in sorted(contents.items())
        },
    }


def refresh_manifest(source: Path, include_datasets: bool | None = None) -> dict[str, Any]:
    """Refresh hashes for approved files in an existing, edited release snapshot."""
    source = source.resolve()
    manifest_path = source / "release_manifest.json"
    if include_datasets is None:
        previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        include_datasets = bool(previous.get("full_synthetic_datasets_included", False))
    selected = select_files(source, include_datasets)
    # Hash the files actually present, not the source of config remapping.
    selected = {relative: _source_file(source, relative) for relative in selected}
    validate_local_imports(source, selected)
    manifest = build_manifest(read_release_contents(selected), include_datasets)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare_release(source: Path, destination: Path, include_datasets: bool = False) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.absolute()
    if destination.is_symlink() or destination.resolve() != destination:
        raise ValueError("Release destination must not traverse symlinks or parent components")
    if destination == source or destination in source.parents:
        raise ValueError("Release destination cannot replace the source repository")
    if source in destination.parents and destination.relative_to(source).parts[0] != "public_release":
        raise ValueError("Use public_release/ for snapshots inside the working repository")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("Release destination already exists and is not empty")
    selected = select_files(source, include_datasets)
    validate_local_imports(source, selected)
    contents = read_release_contents(selected)
    manifest = build_manifest(contents, include_datasets)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".release-export-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "release"
        staging.mkdir()
        for relative, data in contents.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (staging / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.replace(destination)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--destination", type=Path)
    operation.add_argument("--refresh-manifest", action="store_true", help="Update hashes of approved files in this snapshot after edits")
    parser.add_argument("--include-datasets", action="store_true", help="Also stage full English/German exports for review; their redistribution review is pending")
    args = parser.parse_args()
    try:
        source = Path(__file__).resolve().parents[1]
        if args.refresh_manifest:
            manifest = refresh_manifest(source, True if args.include_datasets else None)
        else:
            manifest = prepare_release(source, args.destination, args.include_datasets)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: manifest[key] for key in ("file_count", "payload_bytes", "full_synthetic_datasets_included", "publication_status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
