"""CGSD construction layer.

    from cgsd.constructor import build_arms, feasibility, extinction, gates

⛔ The modules import each other flatly (``from build_arms import ...``) because that is how the
experiment scripts and the data-free gate harnesses load them, and because ``build_arms.py`` is
frozen: its source hash is an input to the calibration signature, so it cannot be edited to suit
a packaging preference. This shim puts the directory on sys.path so both import styles resolve to
the same module objects.
"""
import sys as _sys
from pathlib import Path as _Path

_here = str(_Path(__file__).resolve().parent)
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import build_arms, extinction, feasibility, gates  # noqa: E402,F401

__all__ = ["build_arms", "extinction", "feasibility", "gates"]
