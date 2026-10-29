# Reproduce the release

Run commands from the repository root. The released saved predictions support result reproduction on CPU without videos, model weights, API credentials, or fresh inference.

## 1. Benchmark and held-out tables

```bash
python -m evaluation.reproduce --output-dir results/tables
```

This command scores every main, video-only, baseline, and adapted prediction file against its selected cohort. It checks all **144 main-table F1 values** (24 configurations/settings times six components) against the paper's three-decimal values and writes unrounded counts and scores. The output tables explicitly distinguish the full 1,024-clip cohort from the held-out 197-clip cohort.

The main table uses the original paper-matched Nemotron-Instruct predictions. The older supplement's two mismatched prediction versions are not used. `results/source_manifest.json` records file hashes and archive provenance. The original scoring implementation is retained, including deterministic alignment tie-breaking and the fallback on difficult cases.

## 2. Confidence intervals and additional analyses

```bash
python -m pip install -r requirements.txt
python analysis/compute_results.py
python analysis/frame_controls.py
python analysis/prefill.py
python analysis/oov_bound.py
python analysis/audit.py
python analysis/audit_sampling.py
python analysis/tie_sensitivity.py
```

The bootstrap analysis recomputes counts from saved predictions, then performs 10,000 paired source-video resamples. It produces 720 metric estimates and 164 paired contrasts. Frame/seed controls use the resulting count cache. See [analysis documentation](../analysis/README.md) for seeds, dependencies, output schemas, and the precise interpretation of each analysis.

Useful checks include:

| Result | Expected value |
|:--|:--|
| Gemini-3.1 raw ALL | .692 |
| Qwen3-Thinking raw ALL | .608 |
| Paired raw ALL gap | +.084 [.067, .101] |
| Qwen3-Thinking OOV-only ALL bound | .638934 |
| Original prefill graphs differing from gold | 958/1,023 |
| Gold edges absent from those prefills | 1,655/4,429 |
| Prefill-absent A2A support | 186 edges |
| Gemini-3.1 / Qwen2.5 recall on that subset | 99/186 / 31/186 |
| Gold edge audit unanimity | 106/114 |
| Design-weighted gold audit rate | .9101877633 |

The prefill export uses the original human-annotation inputs. It does not replace an unavailable original response with a later recovered graph. The audit export retains all 330 judgments and includes the sampling information needed to recompute the design-weighted estimate.

The original graph-validation study can also be recomputed:

```bash
python -m evaluation.score_validation
```

It reproduces 224/236 acceptance judgments (94.9%), with four skips excluded, and the two validator-pair Gwet AC1 values (.945 and .861). These shown-graph acceptance judgments are distinct from the later edge audit's unanimity criterion.

## 3. Figures

The four main figures have editable generators, frozen inputs, and real example frames under [figures/](../figures/README.md). Website previews are included in `assets/`. Figure generation is separate from numerical evaluation.

## 4. New inference and training

The [model protocols](model_protocols.md), [inference sources](../inference/), and [training sources](../training/) document the historical runs and portable entry points. Each model requires its own compatible environment and weights. API-backed models additionally require user-supplied credentials and may change over time. Matching the saved results does not imply that a new stochastic generation will be byte-identical.

This release validates numerical reproduction from saved outputs. GPU inference and LoRA training have not been rerun as part of release packaging; their source availability and command validation are documented separately.
