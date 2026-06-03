"""Baseline submission: per-subject mean-accuracy lookup.

Reference baseline for the Predictive AI Evaluation Challenge. Every test
subject already appears in the training matrix, so we predict the subject's
historical mean correctness (shrunk toward the global mean) for any new item,
ignoring item content entirely. This beats the constant-0.5 prior while staying
trivially fast and cheap.

Fit the prior offline with ``train_baseline.py`` (writes ``subject_priors.json``
next to this file), then bundle ``model.py`` + ``subject_priors.json`` in the
submission ZIP. No third-party runtime dependencies; the artifact is plain JSON.

    def predict(input: dict, labeled: list[dict] | None = None) -> float
"""

from __future__ import annotations

import json
from pathlib import Path

_EPS = 1e-4  # keep predictions strictly inside (0, 1) for the log-loss metric
_ARTIFACT = Path(__file__).resolve().parent / "subject_priors.json"

# ---------------------------------------------------------------------------
# Module-level init: runs once when the container starts.
# ---------------------------------------------------------------------------
try:
    _data = json.loads(_ARTIFACT.read_text())
    GLOBAL_MEAN: float = float(_data["global_mean"])
    SUBJECT_PRIORS: dict[str, float] = {k: float(v) for k, v in _data["subjects"].items()}
except (OSError, ValueError, KeyError):
    # Missing/corrupt artifact (e.g. running the smoke test before fitting):
    # fall back to an uninformative prior so predict() still returns valid floats.
    GLOBAL_MEAN = 0.5
    SUBJECT_PRIORS = {}


def _subject_name(subject_content: str) -> str:
    """Extract the display name from the leading 'Name:' line of subject_content."""
    first_line = (subject_content or "").split("\n", 1)[0].strip()
    if first_line[:5].lower() == "name:":
        return first_line[5:].strip()
    return first_line


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(subject answers item correctly) as the subject's mean accuracy."""
    name = _subject_name(input.get("subject_content", ""))
    p = SUBJECT_PRIORS.get(name, GLOBAL_MEAN)
    return float(min(max(p, _EPS), 1.0 - _EPS))
