#!/usr/bin/env python3
"""Offline fit for the baseline submission.

Computes a per-subject mean-accuracy prior from the public training data and
writes it to ``subject_priors.json`` next to ``model.py``. This is the
"per-subject ability lookup" reference baseline from the handbook: because every
test subject already appears in the training matrix, the subject's historical
mean correctness is a reasonable cold-start prediction for any new item.

Run offline (training must not happen inside the submission container):

    python train_baseline.py                 # uses ../../data
    python train_baseline.py --data-dir /path/to/data --pseudo-count 50

Design notes
------------
* Responses are averaged only over benchmarks scored in [0, 1] (binary /
  fractional correctness). Likert benchmarks (mtbench 1-10, ultrafeedback 1-5)
  are skipped automatically because their raw values are not correctness
  probabilities. The rule is "skip any benchmark whose max response > 1", so it
  self-corrects if the dataset changes.
* The per-subject estimate is shrunk toward the global mean with a pseudo-count
  (Laplace-style smoothing). This protects sparsely observed subjects and keeps
  predictions calibrated, which the negative-log-loss metric rewards.
* Priors are keyed by the subject ``display_name`` because the runtime
  ``predict()`` only sees ``subject_content`` (a "Name: <display_name>" string),
  not the internal ``subject_id``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_PATH = Path(__file__).resolve().parent / "subject_priors.json"


def response_files(data_dir: Path) -> list[Path]:
    """Per-benchmark response tables (exclude registry and *_traces tables)."""
    return sorted(
        p
        for p in data_dir.glob("*.parquet")
        if p.name not in REGISTRY_FILES and not p.name.endswith("_traces.parquet")
    )


def fit(data_dir: Path, pseudo_count: float) -> dict:
    # subject_id -> [sum_response, count] over eligible ([0,1]-scored) benchmarks.
    by_subject: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    used, skipped = [], []

    for path in response_files(data_dir):
        bench = path.stem
        table = pq.read_table(path, columns=["subject_id", "response"])
        col = table["response"]
        max_resp = pc.max(col).as_py()
        if max_resp is None or max_resp > 1.0:  # Likert / non-correctness scale
            skipped.append(bench)
            continue
        used.append(bench)

        # Keep finite responses in [0, 1]; group by subject for sum and count.
        mask = pc.and_(pc.is_valid(col), pc.and_(pc.greater_equal(col, 0.0),
                                                 pc.less_equal(col, 1.0)))
        grouped = (
            table.filter(mask)
            .group_by("subject_id")
            .aggregate([("response", "sum"), ("response", "count")])
        )
        for sid, s, n in zip(
            grouped["subject_id"].to_pylist(),
            grouped["response_sum"].to_pylist(),
            grouped["response_count"].to_pylist(),
        ):
            acc = by_subject[sid]
            acc[0] += s or 0.0
            acc[1] += n or 0

    # Map subject_id -> display_name and aggregate (handles id->name collisions).
    subj = pq.read_table(
        data_dir / "subjects.parquet", columns=["subject_id", "display_name"]
    ).to_pylist()
    id_to_name = {r["subject_id"]: (r["display_name"] or r["subject_id"]) for r in subj}

    by_name: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for sid, (s, n) in by_subject.items():
        name = id_to_name.get(sid, sid)
        by_name[name][0] += s
        by_name[name][1] += n

    total_sum = sum(s for s, _ in by_name.values())
    total_n = sum(n for _, n in by_name.values())
    global_mean = total_sum / total_n if total_n else 0.5

    # Shrink each subject toward the global mean with a pseudo-count.
    subjects = {
        name: (s + pseudo_count * global_mean) / (n + pseudo_count)
        for name, (s, n) in by_name.items()
        if n > 0
    }

    return {
        "global_mean": global_mean,
        "pseudo_count": pseudo_count,
        "subjects": subjects,
        "meta": {
            "n_subjects": len(subjects),
            "n_response_rows": total_n,
            "benchmarks_used": used,
            "benchmarks_skipped": skipped,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--pseudo-count", type=float, default=50.0,
                    help="shrinkage strength toward the global mean")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"data dir not found: {args.data_dir}")

    artifact = fit(args.data_dir, args.pseudo_count)
    args.out.write_text(json.dumps(artifact, indent=0))

    m = artifact["meta"]
    print(f"global mean accuracy : {artifact['global_mean']:.4f}")
    print(f"subjects with prior  : {m['n_subjects']}")
    print(f"response rows used   : {m['n_response_rows']:,}")
    print(f"benchmarks used      : {', '.join(m['benchmarks_used'])}")
    print(f"benchmarks skipped   : {', '.join(m['benchmarks_skipped']) or '(none)'}")
    print(f"wrote                : {args.out}")


if __name__ == "__main__":
    main()
