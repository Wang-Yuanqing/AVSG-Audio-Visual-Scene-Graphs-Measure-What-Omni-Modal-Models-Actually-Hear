# Reproduce the paper analyses

These scripts use the released gold and saved predictions. They run on CPU and
do not contact a model API or generate new model outputs. Run from the repository
root with Python 3.10+ and NumPy installed:

```bash
python analysis/compute_results.py
python analysis/frame_controls.py
python analysis/prefill.py
python analysis/oov_bound.py
python analysis/audit.py
python analysis/audit_sampling.py
python analysis/tie_sensitivity.py
```

`frame_controls.py` consumes the count cache created by `compute_results.py`.
The other analyses are independent. Each accepts `--root` to read another
checkout. Results are written under `results/analysis/`.

| Entry point | Inputs | Outputs / paper evidence |
| --- | --- | --- |
| `compute_results.py` | 24 main, 10 video-only and 4 adapted prediction files; gold and official split | 720 component estimates and 164 paired contrasts, including CIs, relation-specific scores, audio-removal gaps and LoRA gap changes |
| `frame_controls.py` | Original Qwen3-thinking predictions plus three files in `inputs/frame_controls/` | 18 frame-budget/seed comparisons with paired intervals |
| `prefill.py` | 1,023 original annotation-input graphs and all 12 raw predictions | Class-level editing counts and model recall on the fixed 186 prefill-absent gold A2A edges |
| `oov_bound.py` | Gold and raw predictions | Qwen3-thinking OOV-only ALL upper bound of 0.638934 at fixed topology and counts |
| `audit.py` | Anonymous answer-level evidence in `data/audit/edges.jsonl` | 106/114 unanimously correct gold edges; 91.0% design-weighted rate; 11/21 unanimously correct unmatched predictions |
| `audit_sampling.py` | Gold, saved predictions, frozen audit tasks and sampling metadata | Recreates sampled clips, model assignment and edge selection; verifies all 135 displayed edges; unmatched corpus totals 2,233 and 1,990 |
| `tie_sensitivity.py` | All 24 main prediction files | The supplement's single deterministic gold-ID reversal check; pooled Node, Edge and ALL counts remain unchanged |

## Statistical protocol

The full cohort contains 1,024 clips from 455 source videos. The held-out cohort
contains 197 clips from 91 videos. `compute_results.py` performs 10,000 source-video
cluster-bootstrap draws with seed 42 using NumPy PCG64. Within a cohort, all models
and conditions use the same resampling weights. It pools TP, predicted and gold
counts before calculating micro scores. It does not average per-clip F1.

The 95% intervals are pointwise percentile intervals, with no multiplicity
correction. They describe source-sampling variability conditional on the saved
outputs, reference graphs and selected data, not decoding or annotation
uncertainty. Undefined precision/recall values are excluded from the corresponding
interval, with finite-draw counts exposed in the CSV. Zero-TP bootstrap intervals
can collapse to zero and are not a bound on future model performance.

Audio sensitivity is the paired score difference AV minus video-only. LoRA gap
change subtracts the base-model gap from the adapted-model gap on the same 197
clips. Differences are computed before rounding. Neither quantity identifies a
causal fraction of relations attributable to hearing.

The original metric is unchanged. Per-relation counting instruments its chosen
full-graph alignment and checks every clip against the original family and pooled
edge TPs. All OOV labels are retained, explicit invalid outputs remain empty, and
no prediction is repaired during analysis. Recorded alignment fallback cases
remain visible in the output manifests.

The generated `bootstrap/manifest.json` records the metric, gold, script and
prediction hashes, software versions, resampling weight hashes and cohort details.
For repeated analysis, `--reuse-counts results/analysis/bootstrap` is optional;
reuse requires matching gold, metric and prediction hashes. The default run
recomputes counts directly from every prediction.

## Original-prefill provenance

`data/prefill_original/prefill_1023.jsonl` contains parsed graphs from the three
original Gemini 3.1 Pro batches confirmed as the input to human annotation. It
does not substitute later recovered graph versions. The original batches contain
1,731 candidates; only the 1,024 released clip IDs are selected. One saved response
is empty, leaving 1,023 comparable graphs. That clip has four gold edges and no
A2A edges, so all 453 gold A2A edges are covered by this analysis.

The export preserves original node/edge content and removes service wrappers.
`provenance.json` records original batch hashes and the exclusion. Changes are
class-level, ignore attributes, and include vocabulary updates. The fixed absent
A2A set is obtained under the full-graph alignment; models are not realigned only
to this subset. These diagnostics do not eliminate shared prefill bias.

## Audit evidence

Eight non-author volunteers supplied 330 judgments on 135 edges from 24 clips
(22 source videos). Seven supplied 45 answers and one supplied 15; every answer
is retained. There are 75 edges with two raters and 60 with three. Three additional
raters covered the same fixed tasks after the initial paired assignment; a third
judgment was not triggered by disagreement.

`correct`, `incorrect` and `uncertain` are stored verdicts. `uncertain` corresponds
to the interface's **Cannot tell** option. Acceptance requires every available
judgment on an edge to be `correct`; uncertainty remains in the denominator.
For gold, the design-weighted rate is the ratio of weighted accepted edges to
weighted audited edges, using inverse clip inclusion probability (12/395 or
12/629). The unweighted fraction is 106/114 = 93.0%; the weighted rate is 91.0%.

The 21 unmatched model edges are distinct from the 114 gold edges. Their sampled
acceptance rates do not correct model F1 or establish graph completeness. The
anonymous release excludes participant names, enrollment credentials, timestamps
and revision histories. Study consent, unpaid participation, independence and
blinding are documented in `data/audit/protocol.json`.
