"""Adaptive labeling: uncertainty acquisition.

The platform reveals the top-K=5 scored candidates per data category and passes
their ground-truth labels to model.predict() as `labeled`, which uses them to
fit a per-round calibration. We request the items the IRT model is least sure
about (uncalibrated probability nearest 0.5): those near the decision boundary
are the most informative for pinning the calibration curve.

Importing `model` reuses its already-loaded encoder and embedding cache, so the
encoding done here is reused by predict(). The score is always finite, so the
platform never falls back to random selection on our account.
"""

from __future__ import annotations

import model


def acquisition_function(input: dict) -> float:
    return -abs(model.raw_irt_prob(input) - 0.5)
