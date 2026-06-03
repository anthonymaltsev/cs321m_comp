# Probe 1 of 3 — encoder + head load

**Question:** does `sentence-transformers/all-MiniLM-L6-v2` + the `ParamHead`
checkpoint load on the hosted runtime, and can the chain run a forward pass?

**Why this probe first:** the leaderboard IRT NLL of −0.63 lands very close
to where the LOBO "IRT no-text" predictor (−0.669) would land after the
regime-easiness adjustment. That is exactly the fallback behavior of the real
submission when `ENCODER = HEAD = None`. This probe pins that branch.

## Mechanism

Module init copies the real submission's load block 1:1 (offline HF env vars,
broad `except Exception`), then runs **one extra forward pass** through the
encoder and head to catch silent runtime issues that would crash the real
submission inside `predict()` rather than at module load.

`predict()` is a constant:

| state | return | expected NLL (y_bar=0.6447) |
|---|---:|---:|
| load + forward OK | 0.70 | **−0.658** |
| anything failed | 0.30 | **−0.903** |

The two values are ~0.25 NLL apart, far outside stochastic eval noise.

## Build and submit

```bash
cd code/probes/01_load
zip -q ../../01_load_probe.zip model.py head.py models.txt \
    stage2_config.json stage2_mlp.pt
python ../../starting_kit/tools/check_submission_zip.py ../../01_load_probe.zip
```

(Local `run_smoke_test.py` will print 0.7 because the smoke-test path
short-circuits the load. That's expected; the probe only signals on the
hosted runtime.)

## Reading the result

| leaderboard NLL | conclusion |
|---|---|
| ≈ **−0.66** | encoder + head load fine. Move to probe 2. |
| ≈ **−0.90** | encoder/head FAILED to load. Root cause found: HF cache miss, dep issue, or HEAD state-dict shape drift. |
| anything else | unexpected, treat as inconclusive and re-submit. |
