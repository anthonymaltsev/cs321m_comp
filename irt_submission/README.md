# IRT 2PL submission — content-amortized cold-start prediction

A 2-parameter-logistic IRT predictor built as the handbook's three-stage PGE
pipeline. It estimates subject abilities and item difficulty/discrimination from
the training matrix, then learns to predict item parameters from item *text* so
it can score brand-new items from unseen benchmarks.

```
P(subject j correct on item i) = sigmoid( a_i * (theta_j - b_i) )
  theta_j  subject ability        (Stage 1, looked up by display name at runtime)
  b_i      item difficulty        (Stage 2a, predicted from item_content)
  a_i      item discrimination    (Stage 2a, predicted from item_content)
```

## Pipeline

| Stage | Script | What it does |
| --- | --- | --- |
| 1 | `fit_irt.py` | Fit the 2PL on the training matrix (full-batch Adam, Gaussian priors). → `subject_abilities.json`, `item_params.npz` |
| 2a-i | `embed_items.py` | Embed each item's text with `all-MiniLM-L6-v2`. → `item_embeddings.npy` |
| 2a-ii | `train_stage2.py` | Regress embeddings → (b, log a) with a small MLP (`head.py`), best-val early stopping. → `stage2_mlp.pt`, `stage2_config.json` |
| 3 | `model.py` | Runtime: θ lookup + predict (b̂, â) from `item_content`, then per-round calibration → probability. |

Shared code: `common.py` (data loading + cache), `irt.py` (2PL fit), `head.py`
(the regressor, imported by both training and runtime), `calib.py` (numpy IRLS
calibration), `labeling.py` (adaptive-labeling acquisition).

### Adaptive labeling (calibration)

The uncalibrated 2PL ranks unseen-benchmark items well but is miscalibrated (a
new benchmark's absolute difficulty level is unknown). The competition reveals
K=5 ground-truth labels per data category to `predict()` as `labeled`.
`calib.py` fits `z_cal = scale·z + bias + offset[benchmark]` on those labels by
ridge-penalized IRLS — a global Platt correction plus per-benchmark offsets
(used only with ≥2 benchmarks and enough both-class labels), with a guard that
falls back to identity unless it beats uncalibrated on the revealed labels.
`labeling.py` requests the most uncertain items (uncalibrated p nearest 0.5),
the most informative for the calibration curve. Calibration runs even with the
platform's default random labels; the acquisition just improves which labels
arrive.

## Reproduce

```bash
cd code/irt_submission              # use ../../.venv/bin/python3
python fit_irt.py                   # Stage 1   (~10 s on MPS)
python embed_items.py               # Stage 2a embeddings (~1 min)
python train_stage2.py              # Stage 2a head
python evaluate_irt.py              # LOBO cold-start evaluation (~1 min)

# validate the submission against the (read-only) starter kit
python ../starting_kit/tools/run_smoke_test.py .
zip -q ../irt_submission.zip model.py head.py calib.py labeling.py models.txt \
    artifacts/subject_abilities.json artifacts/stage2_mlp.pt artifacts/stage2_config.json
python ../starting_kit/tools/check_submission_zip.py ../irt_submission.zip
```

Submission ZIP = `model.py`, `head.py`, `calib.py`, `labeling.py`, `models.txt`,
and the three small artifacts. The encoder is pre-fetched by the platform via
`models.txt` (no runtime download); `model.py` finds artifacts at the ZIP root
or under `artifacts/`.

## Results — leave-one-benchmark-out (cold-start, 4.34M held-out rows)

| Predictor | Neg Log-Loss ↑ | AUC-ROC ↑ |
| --- | --- | --- |
| subject-mean baseline | −0.673 | 0.512 |
| IRT, ability only (no text) | −0.669 | 0.541 |
| IRT + text (uncalibrated) | −0.687 | 0.588 |
| **IRT + text + calib (K=5/bench)** | **−0.663** | **0.588** |
| IRT + text + calib (K=20/bench) | −0.651 | 0.629 |
| IRT + text + calib (K=50/bench) | −0.595 | 0.702 |
| IRT + oracle item params | −0.358 | 0.914 |

Reading: predicting difficulty from text gives a real **ranking** gain on unseen
benchmarks (AUC 0.512 → 0.588). Uncalibrated it loses on **NLL** (right order,
wrong absolute level), but with the competition's K=5 revealed labels the
per-round calibration flips NLL to **−0.663 — above the baseline** while keeping
the AUC edge, and it scales strongly with more labels. The **oracle** row (true
params) reaches AUC 0.914 / NLL −0.358, so the IRT model itself is strong and
the remaining gap is Stage-2a difficulty-prediction error. (K rows simulate the
labeled channel in `evaluate_irt.py`; at the real K=5/category expect ≈ the
global-Platt row, since per-benchmark offsets need more labels.)

## Next steps

1. **Better difficulty map** (the dominant remaining gap → oracle). Stronger
   encoder (`all-mpnet-base-v2`), add `condition`, or train Stage-2a end-to-end
   against the response likelihood instead of MSE to noisy point estimates.
2. **Acquisition.** Try diversity / k-center sampling (handbook §3.6) vs the
   current uncertainty rule to spread the few labels across each benchmark.
3. **Probe panel** for label-free difficulty signal (see project notes).
