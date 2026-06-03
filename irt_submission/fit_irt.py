#!/usr/bin/env python3
"""Stage 1 CLI: fit the 2PL on the full training matrix and save artifacts.

Writes into artifacts/:
  subject_abilities.json   display_name -> theta  (+ "_default": 0.0)
  item_params.npz          b, a, count, item_ids  (aligned, for Stage 2a)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

import common
from irt import fit_2pl, pick_device, prob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    print("loading training matrix ...")
    ds = common.get_dataset()
    print(f"  {len(ds.y):,} responses | {ds.n_subjects} subjects | "
          f"{ds.n_items:,} items | {len(ds.benchmarks)} benchmarks")

    device = args.device or pick_device()
    print(f"fitting 2PL on {device} ...")
    fit = fit_2pl(ds.subj, ds.item, ds.y, ds.n_subjects, ds.n_items,
                  steps=args.steps, lr=args.lr, device=device, verbose=True)

    # In-sample sanity metrics.
    p = prob(fit.theta[ds.subj], fit.a[ds.item], fit.b[ds.item])
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(np.mean(ds.y * np.log(p) + (1 - ds.y) * np.log(1 - p)))
    print(f"in-sample mean log-lik: {ll:.4f}")
    print(f"theta: mean {fit.theta.mean():.3f} sd {fit.theta.std():.3f} | "
          f"b: mean {fit.b.mean():.3f} sd {fit.b.std():.3f} | "
          f"a: mean {fit.a.mean():.3f} sd {fit.a.std():.3f}")

    # Subject abilities keyed by display name (runtime only sees the name).
    names = common.display_names(ds.subject_ids)
    by_name: dict[str, list[float]] = defaultdict(list)
    for name, th in zip(names, fit.theta):
        by_name[name].append(float(th))
    abilities = {name: float(np.mean(v)) for name, v in by_name.items()}
    abilities["_default"] = 0.0

    common.ARTIFACTS.mkdir(exist_ok=True)
    (common.ARTIFACTS / "subject_abilities.json").write_text(json.dumps(abilities))

    count = np.bincount(ds.item, minlength=ds.n_items).astype(np.int32)
    np.savez(common.ARTIFACTS / "item_params.npz",
             b=fit.b, a=fit.a, count=count,
             item_ids=np.array(ds.item_ids, dtype=object))
    print(f"saved abilities ({len(abilities)-1} subjects) and item_params "
          f"({ds.n_items:,} items) to {common.ARTIFACTS}")


if __name__ == "__main__":
    main()
