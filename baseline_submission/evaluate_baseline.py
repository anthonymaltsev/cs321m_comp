#!/usr/bin/env python3
"""Offline evaluation of the per-subject baseline.

Reports the two competition metrics:
  * Neg Log-Loss : mean log-likelihood of the labels (higher is better, <= 0) --
                   this is the leaderboard's primary metric convention.
  * AUC-ROC      : tie-aware Mann-Whitney AUC (higher is better).

Three regimes, scored only on genuine binary outcomes (response in {0, 1}):
  * constant : predict the global mean for every pair (sanity floor).
  * in-sample: the deployed full-fit prior, scored on the data it was fit on
               (optimistic).
  * LOBO     : leave-one-benchmark-out. Refit subject priors on every benchmark
               except the held-out one, then predict it. This mirrors the real
               cold-start regime (test items come from unseen benchmarks) and is
               the honest estimate of leaderboard performance.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EPS = 1e-4  # matches model.py clamping


def response_files(data_dir: Path) -> list[Path]:
    return sorted(
        p for p in data_dir.glob("*.parquet")
        if p.name not in REGISTRY_FILES and not p.name.endswith("_traces.parquet")
    )


def _counts(table, mask) -> dict[str, int]:
    g = table.filter(mask).group_by("subject_id").aggregate([("response", "count")])
    return dict(zip(g["subject_id"].to_pylist(), g["response_count"].to_pylist()))


def _sum_count(table, mask) -> dict[str, tuple[float, int]]:
    g = (table.filter(mask).group_by("subject_id")
         .aggregate([("response", "sum"), ("response", "count")]))
    return {
        sid: (s or 0.0, n or 0)
        for sid, s, n in zip(g["subject_id"].to_pylist(),
                             g["response_sum"].to_pylist(),
                             g["response_count"].to_pylist())
    }


def clamp(p: float) -> float:
    return min(max(p, EPS), 1.0 - EPS)


def nll_auc(buckets: dict[float, list[int]]) -> tuple[float, float, int]:
    """buckets: score -> [n_pos, n_neg]. Returns (mean_loglik, auc, n_rows)."""
    P = sum(b[0] for b in buckets.values())
    N = sum(b[1] for b in buckets.values())
    total = P + N
    if total == 0:
        return float("nan"), float("nan"), 0

    loglik = sum(npos * math.log(p) + nneg * math.log(1.0 - p)
                 for p, (npos, nneg) in buckets.items())
    mean_loglik = loglik / total

    auc = float("nan")
    if P > 0 and N > 0:
        neg_below = 0
        num = 0.0
        for p in sorted(buckets):  # ascending score
            npos, nneg = buckets[p]
            num += npos * (neg_below + 0.5 * nneg)
            neg_below += nneg
        auc = num / (P * N)
    return mean_loglik, auc, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--pseudo-count", type=float, default=50.0)
    args = ap.parse_args()
    c = args.pseudo_count

    # Per benchmark: prior stats over [0,1] responses, and eval pos/neg over {0,1}.
    prior: dict[str, dict[str, tuple[float, int]]] = {}
    pos: dict[str, dict[str, int]] = {}
    neg: dict[str, dict[str, int]] = {}

    for path in response_files(args.data_dir):
        col_t = pq.read_table(path, columns=["subject_id", "response"])
        col = col_t["response"]
        if (pc.max(col).as_py() or 2.0) > 1.0:  # skip Likert (non-[0,1]) scales
            continue
        bench = path.stem
        in01 = pc.and_(pc.is_valid(col),
                       pc.and_(pc.greater_equal(col, 0.0), pc.less_equal(col, 1.0)))
        prior[bench] = _sum_count(col_t, in01)
        pos[bench] = _counts(col_t, pc.equal(col, 1.0))
        neg[bench] = _counts(col_t, pc.equal(col, 0.0))

    benches = list(prior)

    # Global totals across all eligible benchmarks.
    tot_sum: dict[str, float] = defaultdict(float)
    tot_cnt: dict[str, int] = defaultdict(int)
    for b in benches:
        for sid, (s, n) in prior[b].items():
            tot_sum[sid] += s
            tot_cnt[sid] += n
    grand_sum = sum(tot_sum.values())
    grand_cnt = sum(tot_cnt.values())
    global_mean = grand_sum / grand_cnt if grand_cnt else 0.5

    def eval_subjects(b):
        return set(pos[b]) | set(neg[b])

    # --- constant predictor ---
    const_buckets = defaultdict(lambda: [0, 0])
    p_const = clamp(global_mean)
    for b in benches:
        for sid in eval_subjects(b):
            const_buckets[p_const][0] += pos[b].get(sid, 0)
            const_buckets[p_const][1] += neg[b].get(sid, 0)

    # --- in-sample (full fit) ---
    insample = defaultdict(lambda: [0, 0])
    for b in benches:
        for sid in eval_subjects(b):
            s, n = tot_sum.get(sid, 0.0), tot_cnt.get(sid, 0)
            p = clamp((s + c * global_mean) / (n + c)) if n else p_const
            insample[p][0] += pos[b].get(sid, 0)
            insample[p][1] += neg[b].get(sid, 0)

    # --- LOBO ---
    lobo = defaultdict(lambda: [0, 0])
    per_bench = {}
    for b in benches:
        b_sum = sum(s for s, _ in prior[b].values())
        b_cnt = sum(n for _, n in prior[b].values())
        gm_b = ((grand_sum - b_sum) / (grand_cnt - b_cnt)
                if grand_cnt - b_cnt > 0 else global_mean)
        local = defaultdict(lambda: [0, 0])
        for sid in eval_subjects(b):
            ps, pn = prior[b].get(sid, (0.0, 0))
            s_o, n_o = tot_sum.get(sid, 0.0) - ps, tot_cnt.get(sid, 0) - pn
            p = clamp((s_o + c * gm_b) / (n_o + c)) if n_o > 0 else clamp(gm_b)
            np_, nn_ = pos[b].get(sid, 0), neg[b].get(sid, 0)
            lobo[p][0] += np_; lobo[p][1] += nn_
            local[p][0] += np_; local[p][1] += nn_
        per_bench[b] = nll_auc(local)

    print(f"global mean accuracy: {global_mean:.4f}   pseudo-count: {c:g}")
    print(f"{'regime':<12}{'NegLogLoss':>12}{'AUC-ROC':>10}{'eval rows':>12}")
    for name, buckets in [("constant", const_buckets),
                          ("in-sample", insample), ("LOBO", lobo)]:
        ll, auc, n = nll_auc(buckets)
        print(f"{name:<12}{ll:>12.4f}{auc:>10.4f}{n:>12,}")

    print("\nLOBO per held-out benchmark:")
    print(f"{'benchmark':<16}{'NegLogLoss':>12}{'AUC-ROC':>10}{'eval rows':>12}")
    for b in sorted(per_bench, key=lambda k: per_bench[k][2], reverse=True):
        ll, auc, n = per_bench[b]
        auc_s = f"{auc:>10.4f}" if auc == auc else f"{'n/a':>10}"
        print(f"{b:<16}{ll:>12.4f}{auc_s}{n:>12,}")


if __name__ == "__main__":
    main()
