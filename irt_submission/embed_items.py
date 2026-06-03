#!/usr/bin/env python3
"""Stage 2a, part 1: embed item_content for every fitted item (offline).

Encodes the item text with a sentence-transformer and caches the matrix aligned
to the item order in artifacts/item_params.npz. The same encoder is loaded at
runtime in model.py (declared in models.txt), so a brand-new item is embedded
the same way it was during training.
"""

from __future__ import annotations

import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

import common
from irt import pick_device

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LEN = 256


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    params = np.load(common.ARTIFACTS / "item_params.npz", allow_pickle=True)
    item_ids = list(params["item_ids"])
    texts = common.item_texts(item_ids)
    n_empty = sum(1 for t in texts if not t)
    print(f"{len(texts):,} items to embed ({n_empty} with empty content)")

    device = pick_device()
    print(f"loading {args.model} on {device} ...")
    enc = SentenceTransformer(args.model, device=device)
    enc.max_seq_length = MAX_SEQ_LEN

    emb = enc.encode(
        texts, batch_size=args.batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    ).astype(np.float32)
    print("embeddings:", emb.shape)

    out = common.ARTIFACTS / "item_embeddings.npy"
    np.save(out, emb)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
