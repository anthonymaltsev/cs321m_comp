# Baseline submission — per-subject mean accuracy

A minimal, content-free predictor for the Predictive AI Evaluation Challenge.
For a `(subject, item, benchmark, condition)` query it ignores the item and
predicts the subject's **historical mean correctness** on the public training
data, shrunk toward the global mean. This is the "Stage-1 ability lookup"
reference baseline from the handbook: every test subject already appears in the
training matrix, so its average accuracy is a sane cold-start guess for any new
item. It beats the constant-`0.5` prior and is instant to run.

## Files

| File | Role |
| --- | --- |
| `model.py` | Runtime entry point. Loads `subject_priors.json` at import, defines `predict()`. |
| `subject_priors.json` | Fitted artifact: `global_mean` + per-subject prior, keyed by display name. |
| `train_baseline.py` | Offline fit. Reads `../../data`, writes `subject_priors.json`. |

Only `model.py` + `subject_priors.json` are needed at Codabench runtime.

## How it works

- **Fit (offline).** For each per-benchmark response table, average `response`
  per `subject_id` over benchmarks scored in `[0, 1]`. Likert benchmarks
  (`mtbench` 1–10, `ultrafeedback` 1–5) are skipped automatically (any benchmark
  whose max response > 1). Map `subject_id → display_name`, then shrink toward
  the global mean: `p = (sum + c·global) / (n + c)` with pseudo-count `c=50`.
- **Predict (runtime).** Parse the display name from the leading `Name:` line of
  `input["subject_content"]`, look up its prior, fall back to `global_mean` for
  unknown subjects. Output is clamped to `(0, 1)` for the log-loss metric.
- Keying by display name (not `subject_id`) is deliberate: the runtime only
  exposes `subject_content`, not stable IDs. `labeled` is accepted but unused.

Current fit: global mean **0.6445**, **868** subjects, ~4.35M response rows.

## Reproduce

```bash
# from this directory, using the project venv
../../.venv/bin/python3 train_baseline.py            # rewrites subject_priors.json

# validate against the (read-only) starter kit
../../.venv/bin/python3 ../starting_kit/tools/run_smoke_test.py .
zip -q ../baseline_submission.zip model.py subject_priors.json
../../.venv/bin/python3 ../starting_kit/tools/check_submission_zip.py ../baseline_submission.zip
```

## Natural next steps

- **Item difficulty.** Test items are from unseen benchmarks, so a per-item
  lookup can't transfer — learn a map from `item_content` text → difficulty
  (Stage 2a) and combine with subject ability (an IRT / NCF model).
- **Use `condition`.** Fold the test condition into the subject estimate.
- **Adaptive labeling.** Add `labeling.py` and use the revealed `labeled` set to
  Platt-calibrate predictions per round.
