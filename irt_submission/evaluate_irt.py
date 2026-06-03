#!/usr/bin/env python3
"""Leave-one-benchmark-out (LOBO) evaluation of the IRT pipeline.

This is the honest cold-start estimate: each benchmark is held out in turn, the
2PL and the Stage-2a content head are refit on the *other* benchmarks, and the
held-out benchmark's (subject, item) pairs are predicted from item text alone.
It mirrors the competition, where test items come from unseen benchmarks.

Four predictors are scored on identical rows:
  subj-mean   per-subject mean accuracy, shrunk (the baseline, for reference)
  IRT no-text sigmoid(theta - mean train difficulty)   (ability only)
  IRT +text   sigmoid(a_hat * (theta - b_hat)), params predicted from item text
  IRT oracle  sigmoid(a * (theta - b)) using params fit WITH the held-out data
              (upper bound: what +text would reach if it recovered params exactly)

Metrics: NegLogLoss = mean log-likelihood (higher better, <=0); AUC-ROC.
"""

from __future__ import annotations

import argparse

import numpy as np
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--irt-steps", type=int, default=500)
    ap.add_argument("--head-epochs", type=int, default=80)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or pick_device()

    ds = common.get_dataset()
    emb = np.load(common.ARTIFACTS / "item_embeddings.npy")
    assert len(emb) == ds.n_items, "embeddings not aligned with dataset items"
    import torch

    # Global fit (with all data) -> oracle item params and ability.
    print("global IRT fit (oracle params) ...")
    g = fit_2pl(ds.subj, ds.item, ds.y, ds.n_subjects, ds.n_items,
                steps=args.irt_steps, device=device)

    N = len(ds.y)
    pred = {k: np.zeros(N, np.float64) for k in ("subj", "notext", "text", "oracle")}
    z_text = np.zeros(N, np.float64)  # raw text logit, for calibration simulation

    for b_idx, bench in enumerate(ds.benchmarks):
        tr = ds.obs_bench != b_idx
        te = ~tr
        # Stage 1: refit IRT without this benchmark.
        f = fit_2pl(ds.subj[tr], ds.item[tr], ds.y[tr], ds.n_subjects, ds.n_items,
                    steps=args.irt_steps, device=device)

        # subject-mean baseline on training rows.
        gm = float(ds.y[tr].mean())
        ssum = np.bincount(ds.subj[tr], weights=ds.y[tr], minlength=ds.n_subjects)
        scnt = np.bincount(ds.subj[tr], minlength=ds.n_subjects)
        smean = (ssum + SHRINK * gm) / (scnt + SHRINK)

        # Stage 2a: refit content head on training-benchmark items.
        tr_items = np.where(ds.item_bench != b_idx)[0]
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(tr_items))
        n_val = max(1, int(0.1 * len(tr_items)))
        val_items = tr_items[perm[:n_val]]
        fit_items = tr_items[perm[n_val:]]
        count = np.bincount(ds.item[tr], minlength=ds.n_items).astype(np.float32)
        log_a = np.log(np.clip(f.a, A_MIN, A_MAX))
        head, st, _ = fit_head(emb, f.b.astype(np.float32), log_a, count,
                               fit_items, val_items, epochs=args.head_epochs, device=device)

        te_items = np.where(ds.item_bench == b_idx)[0]
        with torch.no_grad():
            out = head(torch.as_tensor(emb[te_items], device=device)).cpu().numpy()
        b_hat = np.zeros(ds.n_items); a_hat = np.ones(ds.n_items)
        b_hat[te_items] = out[:, 0] * st["b_std"] + st["b_mean"]
        a_hat[te_items] = np.clip(np.exp(out[:, 1] * st["la_std"] + st["la_mean"]), A_MIN, A_MAX)
        mean_b_tr = float(f.b[tr_items].mean())

        si, ii = ds.subj[te], ds.item[te]
        zt = a_hat[ii] * (f.theta[si] - b_hat[ii])
        z_text[te] = zt
        pred["subj"][te] = smean[si]
        pred["notext"][te] = sigmoid(f.theta[si] - mean_b_tr)
        pred["text"][te] = sigmoid(zt)
        pred["oracle"][te] = sigmoid(np.clip(g.a[ii], A_MIN, A_MAX) * (g.theta[si] - g.b[ii]))
        ll, auc = metrics(ds.y[te], pred["text"][te])
        print(f"  [{bench:14}] +text NLL={ll:.4f} AUC={auc:.4f}  ({te.sum():,} rows)")

    print(f"\n{'predictor':<14}{'NegLogLoss':>12}{'AUC-ROC':>10}")
    labels = {"subj": "subj-mean", "notext": "IRT no-text",
              "text": "IRT +text", "oracle": "IRT oracle"}
    for k in ("subj", "notext", "text", "oracle"):
        ll, auc = metrics(ds.y, pred[k])
        print(f"{labels[k]:<14}{ll:>12.4f}{auc:>10.4f}")
    print(f"\n(pooled over {N:,} held-out rows; subj-mean reproduces the baseline)")

    # --- adaptive-labeling calibration: reveal K labels per benchmark, fit, apply ---
    bench_all = np.array([ds.benchmarks[bi] for bi in ds.obs_bench])
    print(f"\nCalibrated 'IRT +text' (reveal K real labels per benchmark, fit Platt + per-bench offset):")
    print(f"{'labels/bench':<14}{'NegLogLoss':>12}{'AUC-ROC':>10}{'scale':>8}{'bias':>8}{'#offs':>7}")
    rng = np.random.default_rng(0)
    for K in (5, 20, 50):
        reveal = np.concatenate([
            rng.choice(np.where(ds.obs_bench == bi)[0],
                       size=min(K, int((ds.obs_bench == bi).sum())), replace=False)
            for bi in range(len(ds.benchmarks)) if (ds.obs_bench == bi).any()
        ])
        cal = calib.fit_calibration(z_text[reveal], ds.y[reveal], list(bench_all[reveal]))
        delta_vec = np.array([cal.deltas.get(b, 0.0) for b in bench_all])
        p_cal = sigmoid(cal.scale * z_text + cal.bias + delta_vec)
        keep = np.ones(N, bool); keep[reveal] = False
        ll, auc = metrics(ds.y[keep], p_cal[keep])
        print(f"{K:<14}{ll:>12.4f}{auc:>10.4f}{cal.scale:>8.2f}{cal.bias:>8.2f}{len(cal.deltas):>7d}")


if __name__ == "__main__":
    main()
