#!/usr/bin/env python3
"""Stage 2a, part 2: regress item embeddings -> (difficulty, log-discrimination).

Fits the ParamHead on (embedding_i -> b_i, log a_i) from Stage 1, weighting each
item by sqrt(response count) so well-estimated items dominate, with best-val
early stopping. `fit_head` is reused by evaluate_irt.py for per-fold refits.

Writes:
  artifacts/stage2_mlp.pt       ParamHead state_dict
  artifacts/stage2_config.json  arch dims, target mean/std, encoder id
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

import common
from embed_items import MAX_SEQ_LEN, MODEL_ID
from head import ParamHead
from irt import A_MAX, A_MIN, pick_device


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fit_head(emb, b, log_a, count, train_idx, val_idx, *, epochs=120,
             batch_size=1024, lr=1e-3, patience=25, device=None, seed=0, verbose=False):
    """Fit ParamHead on standardized (b, log a) targets; return (head, stats).

    Target standardization uses only `train_idx` so no fold leakage occurs.
    """
    device = device or pick_device()
    bt, lat = b[train_idx], log_a[train_idx]
    b_mean, b_std = float(bt.mean()), float(bt.std() + 1e-6)
    la_mean, la_std = float(lat.mean()), float(lat.std() + 1e-6)
    targets = np.stack([(b - b_mean) / b_std, (log_a - la_mean) / la_std], axis=1)

    X = torch.as_tensor(emb, device=device)
    T = torch.as_tensor(targets, dtype=torch.float32, device=device)
    W = torch.as_tensor(np.sqrt(count), dtype=torch.float32, device=device)
    tr = torch.as_tensor(np.asarray(train_idx), device=device)
    vt = torch.as_tensor(np.asarray(val_idx), device=device)

    torch.manual_seed(seed)
    head = ParamHead(emb.shape[1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    best_val, best_state, best_ep = float("inf"), None, -1
    for ep in range(epochs):
        head.train()
        order = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss = (W[idx] * F.mse_loss(head(X[idx]), T[idx], reduction="none").sum(1)).mean()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(head(X[vt]), T[vt]).item()
        if val_loss < best_val - 1e-5:
            best_val, best_ep = val_loss, ep
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"  ep {ep:3d}  val_mse={val_loss:.4f}")
        if ep - best_ep >= patience:
            break
    head.load_state_dict(best_state)
    head.eval()
    stats = {"b_mean": b_mean, "b_std": b_std, "la_mean": la_mean, "la_std": la_std}
    return head, stats, best_ep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    emb = np.load(common.ARTIFACTS / "item_embeddings.npy")
    params = np.load(common.ARTIFACTS / "item_params.npz", allow_pickle=True)
    b = params["b"].astype(np.float32)
    a = np.clip(params["a"].astype(np.float32), A_MIN, A_MAX)
    count = params["count"].astype(np.float32)
    log_a = np.log(a)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(emb))
    n_val = int(len(emb) * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    print(f"training Stage-2a head: {len(tr_idx):,} train / {n_val:,} val items, dim {emb.shape[1]}")
    head, stats, best_ep = fit_head(emb, b, log_a, count, tr_idx, val_idx,
                                    epochs=args.epochs, seed=args.seed, verbose=True)

    with torch.no_grad():
        pv = head(torch.as_tensor(emb[val_idx], device=next(head.parameters()).device)).cpu().numpy()
    pb = pv[:, 0] * stats["b_std"] + stats["b_mean"]
    pla = pv[:, 1] * stats["la_std"] + stats["la_mean"]
    print(f"best ep {best_ep} | val corr(b)={pearson(pb, b[val_idx]):.3f}  "
          f"corr(log a)={pearson(pla, log_a[val_idx]):.3f}")

    torch.save(head.state_dict(), common.ARTIFACTS / "stage2_mlp.pt")
    (common.ARTIFACTS / "stage2_config.json").write_text(json.dumps({
        "encoder_id": MODEL_ID, "max_seq_len": MAX_SEQ_LEN, "in_dim": emb.shape[1],
        "a_min": A_MIN, "a_max": A_MAX, **stats,
    }))
    print(f"saved head + config to {common.ARTIFACTS}")


if __name__ == "__main__":
    main()
