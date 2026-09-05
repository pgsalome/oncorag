"""Check the source-checkout runner without relying on the working directory."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_cli(cwd, *args):
    env = os.environ.copy()
    for name in ("PYTHONPATH", "OLLAMA_HOST", "OLLAMA_MODEL"):
        env.pop(name, None)
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_oncorag.py"), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


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
