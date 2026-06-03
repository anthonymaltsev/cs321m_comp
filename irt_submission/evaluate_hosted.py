#!/usr/bin/env python3
"""Hosted-style item-cold-start evaluation.

Mimics the Codabench hidden-eval regime more faithfully than LOBO:

  * the hosted eval samples ~5000 items stratified across data categories and
    asks for predictions on (subject, held-out item) pairs;
  * the items' benchmarks are usually still represented in the participant's
    training matrix (item-cold-start, not benchmark-cold-start).

`evaluate_irt.py` removes whole benchmarks at a time; this script removes a
stratified sample of items and refits Stage 1 (IRT) and Stage 2a (text head) on
the remainder. Adaptive labels are simulated by revealing K labels per
benchmark from the held-out pool and fitting `calib.Calibrator` on them.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import calib
import common
from irt import A_MAX, A_MIN, fit_2pl, pick_device
from train_stage2 import fit_head

SHRINK = 50.0  # subject-mean pseudo-count, matches the baseline


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    auc = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")
    return ll, auc


def stratified_item_holdout(item_bench: np.ndarray, n_benches: int,
                            target_total: int, seed: int) -> np.ndarray:
    """Hold out an equal share of items per benchmark, capped at availability.

    Mirrors 'each data category contributes an approximately equal share' from
    the handbook. Returns global item indices.
    """
    rng = np.random.default_rng(seed)
    per = max(1, target_total // n_benches)
    parts = []
    for b in range(n_benches):
        items_b = np.where(item_bench == b)[0]
        k = min(per, len(items_b))
        if k:
            parts.append(rng.choice(items_b, size=k, replace=False))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden-items", type=int, default=5000,
                    help="Target held-out items, stratified equally by benchmark.")
    ap.add_argument("--irt-steps", type=int, default=500)
    ap.add_argument("--head-epochs", type=int, default=80)
    ap.add_argument("--K", type=int, default=5,
                    help="Revealed labels per benchmark for calibration.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or pick_device()

    ds = common.get_dataset()
    emb = np.load(common.ARTIFACTS / "item_embeddings.npy")
    assert len(emb) == ds.n_items, "embeddings not aligned with dataset items"

    held = stratified_item_holdout(ds.item_bench, len(ds.benchmarks),
                                   args.hidden_items, args.seed)
    is_held = np.zeros(ds.n_items, bool); is_held[held] = True
    tr_rows = ~is_held[ds.item]
    te_rows = ~tr_rows
    print(f"held-out items : {is_held.sum()} / {ds.n_items}")
    print(f"train rows     : {tr_rows.sum():,}")
    print(f"eval rows      : {te_rows.sum():,}")
    print(f"benchmarks     : {len(ds.benchmarks)}")

    # ----- Stage 1: refit IRT on training rows (items not held out) -----
    print("\nfitting 2PL IRT on train rows ...")
    f = fit_2pl(ds.subj[tr_rows], ds.item[tr_rows], ds.y[tr_rows],
                ds.n_subjects, ds.n_items, steps=args.irt_steps, device=device)

    gm = float(ds.y[tr_rows].mean())
    ssum = np.bincount(ds.subj[tr_rows], weights=ds.y[tr_rows], minlength=ds.n_subjects)
    scnt = np.bincount(ds.subj[tr_rows], minlength=ds.n_subjects)
    smean = (ssum + SHRINK * gm) / (scnt + SHRINK)

    # ----- Stage 2a: refit text head on items not held out -----
    tr_items = np.where(~is_held)[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tr_items))
    n_val = max(1, int(0.1 * len(tr_items)))
    val_items = tr_items[perm[:n_val]]
    fit_items = tr_items[perm[n_val:]]
    count = np.bincount(ds.item[tr_rows], minlength=ds.n_items).astype(np.float32)
    log_a = np.log(np.clip(f.a, A_MIN, A_MAX))
    print(f"fitting Stage 2a head on {len(fit_items):,} items "
          f"(val={len(val_items):,}) ...")
    head, st, _ = fit_head(emb, f.b.astype(np.float32), log_a, count,
                           fit_items, val_items, epochs=args.head_epochs,
                           device=device)

    # ----- Predict (b_hat, a_hat) for held-out items from text -----
    with torch.no_grad():
        out = head(torch.as_tensor(emb[held], device=device)).cpu().numpy()
    b_hat = np.zeros(ds.n_items); a_hat = np.ones(ds.n_items)
    b_hat[held] = out[:, 0] * st["b_std"] + st["b_mean"]
    a_hat[held] = np.clip(np.exp(out[:, 1] * st["la_std"] + st["la_mean"]),
                          A_MIN, A_MAX)

    # ----- Score predictors on held-out pairs -----
    te_idx = np.where(te_rows)[0]
    si, ii = ds.subj[te_idx], ds.item[te_idx]
    y_te = ds.y[te_idx]
    bench_te = np.array([ds.benchmarks[b] for b in ds.obs_bench[te_idx]])

    p_subj = smean[si]
    z_text = a_hat[ii] * (f.theta[si] - b_hat[ii])
    p_text = sigmoid(z_text)

    rows = [
        ("subj-mean baseline",         metrics(y_te, p_subj)),
        ("IRT + text (uncalibrated)",  metrics(y_te, p_text)),
    ]

    # ----- Calibration: K labels/benchmark drawn from the held-out pool -----
    rng = np.random.default_rng(args.seed)
    reveal_pos = []  # positions within te_idx, NOT global row indices
    for bi in range(len(ds.benchmarks)):
        pos_b = np.where(ds.obs_bench[te_idx] == bi)[0]
        if len(pos_b) == 0:
            continue
        k = min(args.K, len(pos_b))
        reveal_pos.append(rng.choice(pos_b, size=k, replace=False))
    reveal_pos = np.concatenate(reveal_pos) if reveal_pos else np.empty(0, int)

    if len(reveal_pos):
        cal = calib.fit_calibration(z_text[reveal_pos], y_te[reveal_pos],
                                    list(bench_te[reveal_pos]))
        delta_vec = np.array([cal.deltas.get(b, 0.0) for b in bench_te])
        p_cal = sigmoid(cal.scale * z_text + cal.bias + delta_vec)
        keep = np.ones(len(te_idx), bool); keep[reveal_pos] = False
        rows.append(
            (f"IRT + text + calib (K={args.K}/bench)",
             metrics(y_te[keep], p_cal[keep]))
        )
        print(f"\ncalibration : scale={cal.scale:.3f}, bias={cal.bias:.3f}, "
              f"#deltas={len(cal.deltas)}, labels={len(reveal_pos)}")

    print(f"\n{'predictor':<36}{'NLL':>10}{'AUC':>8}")
    for name, (ll, auc) in rows:
        print(f"{name:<36}{ll:>10.4f}{auc:>8.4f}")
    print(f"\n(pooled over {len(te_idx):,} held-out rows; "
          f"{is_held.sum():,} held-out items stratified across "
          f"{len(ds.benchmarks)} benchmarks)")


if __name__ == "__main__":
    main()
