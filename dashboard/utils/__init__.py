"""Adds the project root to sys.path so `from src.*` imports work.
Any page importing `from src...` must import from `utils` first, or the
src import runs too early and fails (see pages/5_What_If_Simulator.py)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
