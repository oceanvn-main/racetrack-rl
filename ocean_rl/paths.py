"""Stable locations for user maps and experiment output."""
from pathlib import Path
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
MAPS_DIR = Path(__file__).resolve().parent / "data" / "maps"
# Outputs belong to the caller's workspace, never the installed package.
RESULTS_DIR = Path("results")
