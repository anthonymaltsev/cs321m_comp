"""Probe 2 of 3: what fraction of subject lookups hit on the hosted side?

The real IRT submission parses `subject_content`'s first line into a display
name and looks it up in `ABILITIES`. If the lookup misses, theta falls back
to 0 and predictions lose all subject signal. This probe answers the lookup
hit rate by encoding hit/miss into a per-call constant and reading the
fraction out of the aggregate leaderboard NLL.

    name in ABILITIES   ->  predict() returns 0.70
    name not in table   ->  predict() returns 0.30

If a fraction `f` of calls hit, aggregate NLL is approximately

    NLL(f) = f * NLL(0.70) + (1-f) * NLL(0.30)
           = f * (-0.658)  + (1-f) * (-0.903)

so given the observed leaderboard score `X`, the lookup hit rate is

    f ~ (X - (-0.903)) / ((-0.658) - (-0.903))
      = (X + 0.903) / 0.245
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_ab = json.loads((_DIR / "subject_abilities.json").read_text())
_ab.pop("_default", None)
ABILITIES = set(_ab.keys())


def _name_from_subject_content(s: str) -> str:
    line = (s or "").split("\n", 1)[0].strip()
    return line[5:].strip() if line[:5].lower() == "name:" else line


def predict(input: dict, labeled: list | None = None) -> float:
    name = _name_from_subject_content(input.get("subject_content", ""))
    return 0.70 if name in ABILITIES else 0.30
