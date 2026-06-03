# Diagnostic probes — reading the hosted runtime through the NLL channel

The Codabench hosted runtime squashes stdout/stderr and hides tracebacks, so
the only signals that survive back to the participant are:

1. **Submission status** (Finished vs Failed) — one bit per run.
2. **Aggregate NLL** (and AUC) — a continuous number computed from your
   prediction floats and the hidden labels.

These probes turn (2) into a diagnostic channel. Each probe replaces the
real `predict()` with a constant whose value depends on whether some
internal condition holds. The leaderboard NLL then pins which branch fired.

## Reference NLL table at y_bar ≈ 0.6447

| return | NLL |
|---:|---:|
| 0.30 | −0.903 |
| 0.50 | −0.693 |
| 0.65 (optimal const) | −0.651 |
| 0.70 | −0.658 |
| 0.80 | −0.716 |

A probe returning 0.70 when condition `C` holds and 0.30 otherwise will
score somewhere between −0.658 (C always true) and −0.903 (C always false),
linearly in the fraction of calls where C holds.

## Order of submission (highest value first)

| # | Probe | What it answers |
|---|---|---|
| **1** | [`01_load`](01_load/README.md) | Did `sentence-transformers/all-MiniLM-L6-v2` + the head checkpoint load and run a forward pass? |
| **2** | [`02_subject`](02_subject/README.md) | What fraction of hosted calls' subject display names hit our `ABILITIES` table? |
| **3** | [`03_calib`](03_calib/README.md) | When labels are revealed, does the calibrator return a non-identity map, or is it vetoed back to identity? |

Probe 1 first because the leaderboard IRT NLL (−0.63) is suspiciously close
to where the LOBO `IRT no-text` row (−0.669) would project under the easier
hosted regime, which is exactly what the real submission falls back to when
`ENCODER = HEAD = None`. If probe 1 says load is fine, move to probe 2;
otherwise the root cause is found.

## Build all three (from the repo root)

```bash
for p in 01_load 02_subject 03_calib; do
  cd code/probes/$p
  # Each probe's README lists the exact file set; this command is the union.
  zip -q ../../../${p}_probe.zip $(ls *.py *.json *.pt models.txt 2>/dev/null)
  python ../../starting_kit/tools/check_submission_zip.py ../../../${p}_probe.zip
  cd -
done
```

The 50/day per-team submission cap leaves plenty of headroom for these
three plus a re-run if a result lands somewhere unexpected. Probes don't
hurt your leaderboard standing because the grade is taken as the maximum
over all your submissions.
