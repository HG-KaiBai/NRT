"""Reproducible real-world right-censored survival comparisons."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
NRT_ROOT = ROOT.parent
if str(NRT_ROOT) not in sys.path:
    sys.path.insert(0, str(NRT_ROOT))
