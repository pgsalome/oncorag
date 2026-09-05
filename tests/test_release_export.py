import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_export", ROOT / "scripts/prepare_public_release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    files = {
        "README.md": "Public source snapshot\n",
        "pyproject.toml": '[project]\nname = "test-release"\nversion = "0.1"\n',
        "oncoraggraph/__init__.py": "from .core import run\n",
        "oncoraggraph/core.py": "def run():\n    return 1\n",
        "configs/system.public.yaml": "llm_backend: ollama_local\n",
        "oncoraggraph/system_config.yaml": "private local deployment configuration\n",
        "examples/features.synthetic.yaml": "features: []\n",
        "examples/datasets/demo/english/manifest.json": "{}\n",
        "examples/datasets/english/manifest.json": '{"files": {}}\n',
        "examples/datasets/german/manifest.json": '{"files": {}}\n',
        "scripts/run_oncorag.py": "print('runner')\n",
        "configs/oncorag_synthetic_mixed.json": "{}\n",
        "tests/test_public.py": "def test_public():\n    assert True\n",
        ".env": "SECRET=never-copy\n",
        ".git/config": "private repository history\n",
        "analysis/patient_results.json": '{"patient_id": "private"}\n',
        "oncoraggraph/prompt_cache/patient.json": '{"clinical_text": "never-copy"}\n',
        "oncoraggraph/config/feature_configs_real/cohort.json": '{"private": true}\n',
        "oncoraggraph/chat/private_prototype.py": "private prototype\n",
        "oncoraggraph/old_config_scripts/create_config.py": "old script\n",
        "configs/oncorag_full_pipeline_tnbc.json": '{"private": true}\n',
        "scripts/unreviewed.py": "never-copy\n",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(release, "RUNTIME_FILES", ("__init__.py", "core.py"))
    monkeypatch.setattr(release, "REVIEWED_DATASET_MANIFESTS", {
        variant: hashlib.sha256((root / "examples/datasets" / variant / "manifest.json").read_bytes()).hexdigest()
        for variant in ("english", "german")
    })
    return root


def test_clean_export_excludes_clinical_artifacts_and_git_history(source, tmp_path):
    destination = tmp_path / "release"
    manifest = release.prepare_release(source, destination)
    assert (destination / "oncoraggraph/system_config.yaml").read_text() == "llm_backend: ollama_local\n"
    assert (destination / "configs/oncorag_synthetic_mixed.json").is_file()
    assert (destination / "tests/test_public.py").is_file()
    assert (destination / "scripts/run_oncorag.py").read_text() == "print('runner')\n"
    assert "scripts/run_oncorag.py" in manifest["files"]
    assert (destination / "examples/datasets/demo/english/manifest.json").is_file()
    for excluded in (".env", ".git", "analysis", "oncoraggraph/prompt_cache", "oncoraggraph/config/feature_configs_real",
                     "oncoraggraph/chat", "oncoraggraph/old_config_scripts", "configs/oncorag_full_pipeline_tnbc.json",
                     "scripts/unreviewed.py", "examples/datasets/english", "examples/datasets/german"):
        assert not (destination / excluded).exists()
    assert manifest["full_synthetic_datasets_included"] is False
    assert manifest["payload_bytes"] == sum(row["bytes"] for row in manifest["files"].values())
    assert manifest["file_count"] == len(manifest["files"])
    for relative, expected in manifest["files"].items():
        assert hashlib.sha256((destination / relative).read_bytes()).hexdigest() == expected["sha256"]
    assert (source / ".env").read_text() == "SECRET=never-copy\n"


def test_full_cohorts_require_explicit_opt_in(source, tmp_path):
    destination = tmp_path / "release"
    manifest = release.prepare_release(source, destination, include_datasets=True)
    assert (destination / "examples/datasets/english/manifest.json").is_file()
    assert (destination / "examples/datasets/german/manifest.json").is_file()
    assert manifest["full_synthetic_datasets_included"] is True


def test_full_cohort_rejects_unreviewed_manifest(source, tmp_path):
    (source / "examples/datasets/english/manifest.json").write_text('{"files": {}, "changed": true}\n')
    with pytest.raises(ValueError, match="not the reviewed version"):
        release.prepare_release(source, tmp_path / "release", include_datasets=True)
    assert not (tmp_path / "release").exists()


def test_full_cohort_rejects_extra_files(source, tmp_path):
    (source / "examples/datasets/english/extra.txt").write_text("Not reviewed")
    with pytest.raises(ValueError, match="files differ"):
        release.prepare_release(source, tmp_path / "release", include_datasets=True)
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("change", ["changed", "removed"])
def test_full_cohort_rejects_changed_reviewed_payload(source, tmp_path, monkeypatch, change):
    root = source / "examples/datasets/english"
    note = root / "note.txt"
    note.write_bytes(b"Reviewed synthetic text")
    metadata = {"files": {"note.txt": {"sha256": hashlib.sha256(note.read_bytes()).hexdigest(), "bytes": note.stat().st_size}}}
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(metadata))
    monkeypatch.setitem(release.REVIEWED_DATASET_MANIFESTS, "english", hashlib.sha256(manifest.read_bytes()).hexdigest())
    if change == "removed":
        note.unlink()
    else:
        note.write_bytes(b"Unreviewed changed text")
    with pytest.raises(ValueError, match="differs|differ"):
        release.prepare_release(source, tmp_path / "release", include_datasets=True)
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("payload", [
    {"password": "real-secret"},
    {"nested": {"source_style_patient_id": "12345678"}},
    {"nested": [{"api_key": "populated-value"}]},
    {"path": "/" + "home/release-test/private.txt"},
])
def test_metadata_screening_stops_export_before_creating_destination(source, tmp_path, payload):
    (source / "configs/oncorag_synthetic_mixed.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Release screening failed"):
        release.prepare_release(source, tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_credentials_in_python_are_rejected(source, tmp_path):
    (source / "oncoraggraph/core.py").write_text('api_key = "' + "sk-" + "a" * 30 + '"\n')
    with pytest.raises(ValueError, match="Credential marker"):
        release.prepare_release(source, tmp_path / "release")


def test_nonempty_destination_is_not_modified(source, tmp_path):
    destination = tmp_path / "release"
    destination.mkdir()
    existing = destination / "user-file"
    existing.write_text("preserve me")
    with pytest.raises(FileExistsError):
        release.prepare_release(source, destination)
    assert existing.read_text() == "preserve me"


def test_empty_destination_is_supported(source, tmp_path):
    destination = tmp_path / "release"
    destination.mkdir()
    release.prepare_release(source, destination)
    assert (destination / "release_manifest.json").is_file()


def test_symlink_escape_is_rejected(source, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("private external file")
    core = source / "oncoraggraph/core.py"
    core.unlink()
    core.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink escapes"):
        release.prepare_release(source, tmp_path / "release")


def test_missing_runtime_dependency_is_rejected(source, tmp_path):
    (source / "oncoraggraph/core.py").write_text("from .private_runtime import value\n")
    (source / "oncoraggraph/private_runtime.py").write_text("value = 1\n")
    with pytest.raises(ValueError, match="Runtime import missing"):
        release.prepare_release(source, tmp_path / "release")


def test_public_snapshot_cannot_be_written_into_runtime_package(source):
    with pytest.raises(ValueError, match="public_release"):
        release.prepare_release(source, source / "oncoraggraph/release")


def add_public_chat(source):
    files = {
        "oncoraggraph/chat_runtime.py": "from .chat.service import answer\n",
        "oncoraggraph/chat/service.py": "def answer():\n    return 'answer'\n",
        "oncoraggraph/chat/__init__.py": "",
        "oncoraggraph/chat_app.py": "from .chat_runtime import answer\n",
        "run_chatbot.py": "from oncoraggraph.chat_runtime import main\n",
        "streamlit_app.py": "from oncoraggraph.chat_app import main\n",
        "scripts/run_chat_smoke.py": "print('smoke')\n",
    }
    for relative, text in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_public_chat_modules_and_entrypoints_are_included(source, tmp_path):
    add_public_chat(source)
    destination = tmp_path / "release"
    manifest = release.prepare_release(source, destination)
    for relative in ("oncoraggraph/chat_runtime.py", "oncoraggraph/chat/service.py", "oncoraggraph/chat_app.py",
                     "run_chatbot.py", "streamlit_app.py", "scripts/run_chat_smoke.py"):
        assert relative in manifest["files"]
    assert not (destination / "oncoraggraph/chat/private_prototype.py").exists()


def test_legacy_chat_entrypoint_is_rejected(source, tmp_path):
    add_public_chat(source)
    (source / "run_chatbot.py").write_text("from private_checkout import old_chatbot\n")
    with pytest.raises(ValueError, match="portable chat runtime"):
        release.prepare_release(source, tmp_path / "release")


def test_refresh_updates_actual_files_and_preserves_dataset_opt_in(source, tmp_path):
    destination = tmp_path / "release"
    release.prepare_release(source, destination, include_datasets=True)
    (destination / "oncoraggraph/core.py").write_text("def run():\n    return 2\n")
    (destination / "oncoraggraph/system_config.yaml").write_text("llm_backend: portable_changed\n")
    (destination / "outputs").mkdir()
    (destination / "outputs/patient.json").write_text("{}\n")
    manifest = release.refresh_manifest(destination)
    assert manifest["full_synthetic_datasets_included"] is True
    assert "outputs/patient.json" not in manifest["files"]
    for relative, item in manifest["files"].items():
        assert item["sha256"] == hashlib.sha256((destination / relative).read_bytes()).hexdigest()


def test_refresh_rejects_new_sensitive_content_without_changing_manifest(source, tmp_path):
    destination = tmp_path / "release"
    release.prepare_release(source, destination)
    old_manifest = (destination / "release_manifest.json").read_bytes()
    (destination / "configs/oncorag_synthetic_mixed.json").write_text(json.dumps({"password": "not-public"}))
    with pytest.raises(ValueError, match="screening failed"):
        release.refresh_manifest(destination)
    assert (destination / "release_manifest.json").read_bytes() == old_manifest


def test_reviewed_asset_is_copied_hashed_and_preserved_on_refresh(source, tmp_path, monkeypatch):
    asset = b"\x89PNG\r\n\x1a\nreviewed test asset"
    digest = hashlib.sha256(asset).hexdigest()
    monkeypatch.setattr(release, "REVIEWED_ASSETS", {"graphicalabstract.png": digest})
    (source / "graphicalabstract.png").write_bytes(asset)
    destination = tmp_path / "release"
    manifest = release.prepare_release(source, destination)
    expected = {"sha256": digest, "bytes": len(asset)}
    assert (destination / "graphicalabstract.png").read_bytes() == asset
    assert manifest["files"]["graphicalabstract.png"] == expected
    assert release.refresh_manifest(destination)["files"]["graphicalabstract.png"] == expected


def test_changed_reviewed_asset_is_rejected_before_creating_destination(source, tmp_path, monkeypatch):
    original = b"\x89PNG\r\n\x1a\nreviewed test asset"
    monkeypatch.setattr(release, "REVIEWED_ASSETS", {"graphicalabstract.png": hashlib.sha256(original).hexdigest()})
    (source / "graphicalabstract.png").write_bytes(original + b"unreviewed metadata")
    destination = tmp_path / "not-created" / "release"
    with pytest.raises(ValueError, match="approved SHA-256"):
        release.prepare_release(source, destination)
    assert not destination.parent.exists()


def test_refresh_rejects_changed_asset_without_modifying_manifest(source, tmp_path, monkeypatch):
    asset = b"\x89PNG\r\n\x1a\nreviewed test asset"
    monkeypatch.setattr(release, "REVIEWED_ASSETS", {"graphicalabstract.png": hashlib.sha256(asset).hexdigest()})
    (source / "graphicalabstract.png").write_bytes(asset)
    destination = tmp_path / "release"
    release.prepare_release(source, destination)
    previous = (destination / "release_manifest.json").read_bytes()
    (destination / "graphicalabstract.png").write_bytes(asset + b"changed")
    with pytest.raises(ValueError, match="approved SHA-256"):
        release.refresh_manifest(destination)
    assert (destination / "release_manifest.json").read_bytes() == previous


def test_unlisted_binary_remains_excluded_and_rejected(source, tmp_path):
    asset = b"\x89PNG\r\n\x1a\nunreviewed image"
    (source / "unreviewed.png").write_bytes(asset)
    destination = tmp_path / "release"
    manifest = release.prepare_release(source, destination)
    assert "unreviewed.png" not in manifest["files"]
    assert not (destination / "unreviewed.png").exists()
    with pytest.raises(ValueError, match="Unexpected non-text"):
        release.validate_file("unreviewed.png", asset)
    with pytest.raises(ValueError, match="Unexpected non-text"):
        release.validate_file("README.md", asset)


def test_actual_graphical_abstract_matches_reviewed_pin():
    asset = (ROOT / "graphicalabstract.png").read_bytes()
    assert asset.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(asset) == 493926
    assert hashlib.sha256(asset).hexdigest() == release.REVIEWED_ASSETS["graphicalabstract.png"]
    release.validate_file("graphicalabstract.png", asset)
    with pytest.raises(ValueError, match="approved SHA-256"):
        release.validate_file("graphicalabstract.png", asset + b"changed")
    with pytest.raises(ValueError, match="Unexpected non-text"):
        release.validate_file("renamed.png", asset)
