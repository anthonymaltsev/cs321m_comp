# Probe 3 of 3 — calibration applied vs vetoed

**Question:** when the platform reveals `labeled` to `predict()`, does
`calib.fit_calibration` return a non-identity map, or does the validation
guard veto it back to identity?

**Why this probe third:** only worth running once probes 1 and 2 pass.
If the encoder didn't load or the subject lookup misses, calibration isn't
the bottleneck. If both upstream stages work, this isolates whether the
adaptive-labeling channel is contributing.

## Mechanism

Module init replicates the real IRT submission (encoder + head + abilities).
On the first `predict()` call, the probe inspects the `labeled` argument the
platform passed, runs `calib.fit_calibration` exactly as the real submission
would, and records the outcome:

| state | `predict()` returns | expected NLL |
|---|---:|---:|
| labeled present, calib non-identity | 0.70 | −0.658 |
| labeled present, calib vetoed to identity | 0.30 | −0.903 |
| labeled empty / error | 0.50 | −0.693 |

Subsequent calls return the cached probe value, so the aggregate NLL is one
of the three constants above.

## Build and submit

```bash
cd code/probes/03_calib
zip -q ../../03_calib_probe.zip model.py head.py calib.py models.txt \
    stage2_config.json stage2_mlp.pt subject_abilities.json
python ../../starting_kit/tools/check_submission_zip.py ../../03_calib_probe.zip
```

## Reading the result

| leaderboard NLL | conclusion |
|---|---|
| ≈ −0.66 | calibration is applied. The real submission's calibration is doing work; the −0.63 leaderboard score is not explained by a calibration veto. |
| ≈ −0.90 | calibration is being vetoed every round. Try widening `min_per_bench` thresholds, dropping the validation guard, or fitting only `scale` (drop the per-benchmark offsets and global bias). |
| ≈ −0.69 | labeled was empty. Adaptive-labeling channel isn't supplying labels for some reason; the real submission's calibration never had a chance. |
