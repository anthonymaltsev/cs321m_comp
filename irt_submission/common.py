"""Shared data loading for the IRT pipeline (offline only).

Loads the public training matrix into compact integer-indexed arrays:
binary responses (subject, item) -> y in {0,1}, plus the lookups the later
stages need (item text for Stage 2a, subject display name for the runtime θ
lookup, benchmark id per observation for leave-one-benchmark-out evaluation).

Only benchmarks scored in [0,1] are kept, and only rows with response in {0,1}
(genuine binary outcomes, matching the hidden binary labels). Likert benchmarks
(mtbench, ultrafeedback) are skipped automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
_CACHE = ARTIFACTS / "dataset_cache.npz"


@dataclass
class Dataset:
    subj: np.ndarray          # int32 [N]  subject index per observation
    item: np.ndarray          # int32 [N]  item index per observation
    y: np.ndarray             # float32 [N] binary response
    obs_bench: np.ndarray     # int32 [N]  benchmark index per observation
    subject_ids: list[str]    # index -> subject_id
    item_ids: list[str]       # index -> item_id
    benchmarks: list[str]     # index -> benchmark name (file stem)
    item_bench: np.ndarray    # int32 [n_items] benchmark index per item

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def response_files(data_dir: Path) -> list[Path]:
    return sorted(
        p for p in data_dir.glob("*.parquet")
        if p.name not in REGISTRY_FILES and not p.name.endswith("_traces.parquet")
    )


def load_responses(data_dir: Path = DATA_DIR) -> Dataset:
    subj_map: dict[str, int] = {}
    item_map: dict[str, int] = {}
    subj_codes: list[int] = []
    item_codes: list[int] = []
    ys: list[float] = []
    obs_bench: list[int] = []
    benchmarks: list[str] = []
    item_bench_d: dict[int, int] = {}

    for path in response_files(data_dir):
        table = pq.read_table(path, columns=["subject_id", "item_id", "response"])
        col = table["response"]
        if (pc.max(col).as_py() or 2.0) > 1.0:  # Likert / non-[0,1] -> skip
            continue
        binary = pc.or_(pc.equal(col, 0.0), pc.equal(col, 1.0))
        table = table.filter(binary)
        if table.num_rows == 0:
            continue

        b_idx = len(benchmarks)
        benchmarks.append(path.stem)
        sids = table["subject_id"].to_pylist()
        iids = table["item_id"].to_pylist()
        rsp = table["response"].to_pylist()
        for s, it, r in zip(sids, iids, rsp):
            sc = subj_map.setdefault(s, len(subj_map))
            ic = item_map.get(it)
            if ic is None:
                ic = item_map[it] = len(item_map)
                item_bench_d[ic] = b_idx
            subj_codes.append(sc)
            item_codes.append(ic)
            ys.append(r)
            obs_bench.append(b_idx)

    subject_ids = [None] * len(subj_map)
    for s, i in subj_map.items():
        subject_ids[i] = s
    item_ids = [None] * len(item_map)
    for it, i in item_map.items():
        item_ids[i] = it
    item_bench = np.zeros(len(item_ids), dtype=np.int32)
    for ic, bc in item_bench_d.items():
        item_bench[ic] = bc

    return Dataset(
        subj=np.asarray(subj_codes, dtype=np.int32),
        item=np.asarray(item_codes, dtype=np.int32),
        y=np.asarray(ys, dtype=np.float32),
        obs_bench=np.asarray(obs_bench, dtype=np.int32),
        subject_ids=subject_ids,
        item_ids=item_ids,
        benchmarks=benchmarks,
        item_bench=item_bench,
    )


def get_dataset(data_dir: Path = DATA_DIR, use_cache: bool = True) -> Dataset:
    """Load responses, caching the parsed arrays/maps to disk for fast reuse."""
    if use_cache and _CACHE.exists():
        z = np.load(_CACHE, allow_pickle=False)
        meta = json.loads((_CACHE.with_suffix(".meta.json")).read_text())
        return Dataset(
            subj=z["subj"], item=z["item"], y=z["y"], obs_bench=z["obs_bench"],
            item_bench=z["item_bench"], subject_ids=meta["subject_ids"],
            item_ids=meta["item_ids"], benchmarks=meta["benchmarks"],
        )
    ds = load_responses(data_dir)
    ARTIFACTS.mkdir(exist_ok=True)
    np.savez(_CACHE, subj=ds.subj, item=ds.item, y=ds.y,
             obs_bench=ds.obs_bench, item_bench=ds.item_bench)
    _CACHE.with_suffix(".meta.json").write_text(json.dumps(
        {"subject_ids": ds.subject_ids, "item_ids": ds.item_ids,
         "benchmarks": ds.benchmarks}))
    return ds


def item_texts(item_ids: list[str], data_dir: Path = DATA_DIR) -> list[str]:
    """Content string per item index (empty string if missing)."""
    tbl = pq.read_table(data_dir / "items.parquet", columns=["item_id", "content"])
    content = dict(zip(tbl["item_id"].to_pylist(), tbl["content"].to_pylist()))
    return [content.get(i) or "" for i in item_ids]


def display_names(subject_ids: list[str], data_dir: Path = DATA_DIR) -> list[str]:
    """display_name per subject index (falls back to subject_id)."""
    tbl = pq.read_table(data_dir / "subjects.parquet",
                        columns=["subject_id", "display_name"])
    name = dict(zip(tbl["subject_id"].to_pylist(), tbl["display_name"].to_pylist()))
    return [name.get(s) or s for s in subject_ids]
