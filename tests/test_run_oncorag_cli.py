"""Check package installation and CLI entry points."""

import importlib
from importlib.metadata import distribution
from importlib.resources import files
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_cli(cwd, *args, module=None):
    env = os.environ.copy()
    for name in ("PYTHONPATH", "OLLAMA_HOST", "OLLAMA_MODEL"):
        env.pop(name, None)
    command = (
        [sys.executable, "-m", module, *args]
        if module
        else [sys.executable, str(ROOT / "scripts/run_oncorag.py"), *args]
    )
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_installed_project_name_and_console_entry_points():
    package = distribution("oncorag")
    assert package.metadata["Name"] == "oncorag"
    entry_points = {
        entry.name: entry
        for entry in package.entry_points
        if entry.group == "console_scripts"
    }
    expected = {
        "oncorag": "oncorag.pipeline:main",
        "oncorag-chat": "oncorag.chat_runtime:main",
    }
    assert {name: entry.value for name, entry in entry_points.items()} == expected
    for name, target in expected.items():
        module, function = target.split(":")
        assert entry_points[name].load() is getattr(importlib.import_module(module), function)


def test_package_import_and_configuration_resource():
    package = importlib.import_module("oncorag")
    assert Path(package.__file__).resolve() == ROOT / "oncorag" / "__init__.py"
    assert files(package).joinpath("system_config.yaml").is_file()


@pytest.mark.parametrize("module", ["oncorag.pipeline", "oncorag.chat_runtime"])
def test_installed_module_help(tmp_path, module):
    result = run_cli(tmp_path, "--help", module=module)

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_installed_pipeline_module_validates_small_cohort(tmp_path):
    result = run_cli(
        tmp_path,
        "--config", str(ROOT / "configs/oncorag_synthetic_mixed.json"),
        "--stage", "validate",
        module="oncorag.pipeline",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"patients": 3, "notes": 9, "features": 4}


def test_source_checkout_runner_help(tmp_path):
    result = run_cli(tmp_path, "--help")

    assert result.returncode == 0, result.stderr
    assert "usage: run_oncorag.py" in result.stdout
    assert "--config" in result.stdout
    assert "--stage {validate,config,graph,extract}" in result.stdout


def test_source_checkout_runner_validates_synthetic_fixture(tmp_path):
    result = run_cli(
        tmp_path,
        "--config", str(ROOT / "configs/oncorag_synthetic_mixed.json"),
        "--stage", "validate",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"patients": 3, "notes": 9, "features": 4}
