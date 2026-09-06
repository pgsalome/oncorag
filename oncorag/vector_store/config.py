"""Load vector-store settings without connecting to a database."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


def validate_vector_store_config(config: Any) -> dict:
    if not isinstance(config, Mapping):
        raise ValueError("vector_store must be an object")
    settings = dict(config)
    backend = settings.setdefault("backend", "chroma")
    if backend not in ("chroma", "iris"):
        raise ValueError("vector_store.backend must be 'chroma' or 'iris'")
    namespace = settings.setdefault("collection_namespace", "default")
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("vector_store.collection_namespace must be a nonempty string")
    for name in ("chroma", "iris"):
        section = settings.setdefault(name, {})
        if not isinstance(section, Mapping):
            raise ValueError(f"vector_store.{name} must be an object")
        settings[name] = dict(section)
    cache_path = settings["chroma"].get("path")
    if cache_path is not None and (not isinstance(cache_path, str) or not cache_path.strip()):
        raise ValueError("vector_store.chroma.path must be a nonempty string")
    if backend == "iris" or settings["iris"]:
        from .iris import validate_iris_config

        settings["iris"] = validate_iris_config(settings["iris"])
    return settings


def load_vector_store_config(
    path: str | Path | None = None, *, backend: str | None = None
) -> dict:
    """Read a vector-store file or the vector_store section of a run config.

    Explicit arguments override environment selections, then system_config.yaml.
    Relative Chroma paths are resolved relative to the configuration file.
    """
    selected_path = path or os.getenv("ONCORAG_VECTOR_STORE_CONFIG")
    config_path = (
        Path(selected_path).expanduser().resolve()
        if selected_path
        else Path(__file__).resolve().parents[1] / "system_config.yaml"
    )
    if not config_path.exists() and not selected_path:
        data = {}
    else:
        contents = config_path.read_text(encoding="utf-8")
        data = json.loads(contents) if config_path.suffix.lower() == ".json" else yaml.safe_load(contents)
    if not isinstance(data, Mapping):
        raise ValueError("Vector-store configuration file must contain an object")
    if "vector_store" in data:
        settings = data["vector_store"]
    elif "backend" in data:
        settings = data
    elif selected_path:
        raise ValueError("Configuration file must include a vector_store object or backend")
    else:
        settings = {}
    if not isinstance(settings, Mapping):
        raise ValueError("vector_store must be an object")
    settings = dict(settings)
    selected_backend = backend or os.getenv("ONCORAG_VECTOR_BACKEND")
    if selected_backend:
        settings["backend"] = selected_backend
    settings = validate_vector_store_config(settings)
    cache_path = settings["chroma"].get("path")
    if cache_path:
        resolved = Path(cache_path).expanduser()
        if not resolved.is_absolute():
            resolved = config_path.parent / resolved
        settings["chroma"]["path"] = str(resolved.resolve())
    return settings
