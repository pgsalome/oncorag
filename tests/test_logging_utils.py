"""Logging remains usable without an undeclared third-party dependency."""

import importlib.util
from pathlib import Path
import sys


def test_logging_import_and_controls_work_without_loguru(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "loguru", None)
    path = Path(__file__).resolve().parents[1] / "oncoraggraph/utils/logging_utils.py"
    spec = importlib.util.spec_from_file_location("isolated_logging_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.log("visible")
    assert "visible" in capsys.readouterr().out
    module.log("debug hidden", debug=True)
    assert capsys.readouterr().out == ""
    module.set_debug_mode(True)
    module.log("debug visible", debug=True)
    assert "debug visible" in capsys.readouterr().out
    module.set_quiet_mode(True)
    module.log("quiet hidden")
    assert capsys.readouterr().out == ""
    module.log("warning visible", level="WARNING")
    assert "warning visible" in capsys.readouterr().out
