"""Run the portable structured-feature workflow from a source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oncoraggraph.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
