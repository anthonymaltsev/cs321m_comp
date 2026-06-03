"""Per-round calibration from revealed labels (numpy only, no torch).

The IRT logit z = a*(theta - b) ranks items well on unseen benchmarks but its
absolute level is off (a new benchmark's overall difficulty is unknown), which
hurts log-loss. Given a handful of ground-truth labels revealed by the adaptive
channel, we fit a calibrated logit

    z_cal = scale * z + bias + delta[benchmark]

by ridge-penalized logistic regression (IRLS). `scale`/`bias` are a global
Platt correction estimated from all revealed labels pooled; `delta[benchmark]`
is a per-benchmark offset, used only for benchmarks with enough both-class
labels and shrunk toward 0 otherwise. Falls back to identity when labels are
too few or single-class, so it can never make an uncalibrated prediction worse
by much.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

SCALE_CLAMP = (0.2, 5.0)


class Calibrator:
    def __init__(self, scale: float = 1.0, bias: float = 0.0, deltas: dict | None = None):
        self.scale = float(np.clip(scale, *SCALE_CLAMP))
        self.bias = float(bias)
        self.deltas = deltas or {}

    @property
    def is_identity(self) -> bool:
        return self.scale == 1.0 and self.bias == 0.0 and not self.deltas

    def apply(self, z, benchmark: str = ""):
        return self.scale * np.asarray(z, float) + self.bias + self.deltas.get(benchmark, 0.0)


def _logloss(z, y):
    p = np.clip(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))), 1e-6, 1 - 1e-6)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def _irls(X, y, pen, w0, iters=60):
    w = w0.astype(float).copy()
    jitter = 1e-6 * np.eye(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        Wd = p * (1 - p) + 1e-6
        g = X.T @ (p - y) + pen * (w - w0)
        H = X.T @ (X * Wd[:, None]) + np.diag(pen) + jitter
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w = w - np.clip(step, -10, 10)  # damp to avoid divergence
        if np.max(np.abs(step)) < 1e-7:
            break
    return w


def fit_calibration(z, y, benchmark, *, min_total=8, min_per_bench=12,
                    reg_delta=1.0, reg_scale=0.2, reg_bias=0.05) -> Calibrator:
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    benchmark = list(benchmark)
    if len(y) < min_total or y.min() == y.max():
        return Calibrator()  # not enough signal -> identity

    by_bench = defaultdict(list)
    for i, b in enumerate(benchmark):
        by_bench[b].append(i)
    # Per-benchmark offsets are only identifiable with >=2 distinct benchmarks
    # (otherwise the offset is collinear with the global bias).
    eligible = []
    if len(by_bench) >= 2:
        eligible = [b for b, ii in by_bench.items()
                    if len(ii) >= min_per_bench and 0 < y[np.array(ii)].mean() < 1]

    cols = [z, np.ones_like(z)]                       # scale, bias
    for b in eligible:
        cols.append(np.array([1.0 if bb == b else 0.0 for bb in benchmark]))
    X = np.stack(cols, axis=1)

    pen = np.full(X.shape[1], reg_delta); pen[0] = reg_scale; pen[1] = reg_bias
    w0 = np.zeros(X.shape[1]); w0[0] = 1.0            # prior: identity
    w = _irls(X, y, pen, w0)
    if not np.all(np.isfinite(w)):
        return Calibrator()

    cal = Calibrator(
        scale=float(w[0]),
        bias=float(np.clip(w[1], -10, 10)),
        deltas={b: float(np.clip(w[2 + k], -10, 10)) for k, b in enumerate(eligible)},
    )
    # Safety net: only keep calibration if it beats identity on the revealed labels.
    delta_vec = np.array([cal.deltas.get(b, 0.0) for b in benchmark])
    if _logloss(cal.scale * z + cal.bias + delta_vec, y) >= _logloss(z, y):
        return Calibrator()
    return cal
