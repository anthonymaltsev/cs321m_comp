# Probe 2 of 3 — subject display-name lookup hit rate

**Question:** for how many of the hosted test calls does the first-line
display-name parse hit our `ABILITIES` table?

**Why this probe second:** if hits are universal we keep the lookup logic as
is; if many miss, predictions collapse to `theta=0` and the IRT predictor
degrades regardless of what the encoder does. This is the second prime
suspect after the encoder load.

## Mechanism

For each call, parse the first line of `subject_content`, apply the same
"Name: ..." stripping the real submission uses, and look up in
`subject_abilities.json`. Return 0.70 on hit, 0.30 on miss.

The aggregate NLL is a convex combination:

```
NLL(f) ~ f * (-0.658) + (1-f) * (-0.903)
```

where `f` is the hit fraction. Back out `f` from the leaderboard score:

```
f ~ (NLL_leaderboard + 0.903) / 0.245
```

| leaderboard NLL | hit rate | conclusion |
|---|---:|---|
| ≈ −0.66 | ~100% | display names match. Move to probe 3. |
| ≈ −0.78 | ~50% | half the subjects are unknown. Likely a formatting drift. |
| ≈ −0.90 | ~0% | no lookups hit. The subject_content first line isn't `Name: <display_name>` as we assume. Major root cause. |

This probe deliberately ships **no** `models.txt`. We don't need the encoder
to test the lookup, and it routes faster.

## Build and submit

```bash
cd code/probes/02_subject
zip -q ../../02_subject_probe.zip model.py subject_abilities.json
python ../../starting_kit/tools/check_submission_zip.py ../../02_subject_probe.zip
```
