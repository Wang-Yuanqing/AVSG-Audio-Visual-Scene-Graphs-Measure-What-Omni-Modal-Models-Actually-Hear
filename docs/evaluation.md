# Evaluation

## Prediction format

Write one JSON object per clip, using the same `filename` as gold:

```json
{"filename":"clip.mp4","nodes":[{"id":"v1","class":"man","attribute":"person"},{"id":"a1","class":"male_speech","attribute":"speaking"}],"edges":[{"source":"v1","edge":"speaking","target":"a1"}],"valid":true}
```

Gold uses `ground_truth_nodes` and `ground_truth_edges`; predictions use `nodes` and `edges`. The metric normalizes both shapes. Predictions should retain free-text attributes for inspection, although scoring ignores them.

Use the archived prompt in `evaluation/prompt.py` to reproduce the historical task specification. It contains three worked examples; it is not a zero-example prompt. Its wording is preserved even where later manuscript terminology is more precise. Out-of-vocabulary labels remain in predictions and affect precision.

## Commands

```bash
# Untuned full-corpus evaluation.
python -m evaluation.score --predictions results/predictions/main/gemini-3.1-pro_raw.jsonl --split full

# Your model on the official held-out split.
python -m evaluation.score --predictions path/to/test_predictions.jsonl --split test

# Reproduce full-corpus and held-out result tables.
python -m evaluation.reproduce --output-dir results/tables
```

Use the cohort-aware wrapper for normal evaluation. The archived metric's direct CLI is retained for provenance, but choosing the correct input cohort is the caller's responsibility when invoking it directly. See the wrapper's `--help` for coverage checks and output options.

## Scoring

**Nodes:** multiset overlap of `(modality, class)` pairs. Attributes and arbitrary IDs do not contribute to matches.

**Edges:** find a class- and modality-respecting partial injective alignment that maximizes the number of matched edges. Matching uses relation labels and mapped endpoints. The three symmetric relations match in either direction. Duplicate edges are treated as occurrences, consuming reference matches at most once.

**Families:** V2V, V2A, and A2A are reported under the same global edge-maximizing alignment. They are not optimized separately. Ties are resolved by the archived implementation's deterministic ordering. A documented combination cap uses a deterministic fallback on difficult graphs; the scorer is preserved rather than silently replaced.

**Aggregation:** headline scores are micro-F1, pooling true positives, prediction counts, and reference counts across clips before calculating F1. Edge pools all three families; ALL pools nodes and edges. Macro-F1, where reported, averages eligible per-clip scores and excludes empty/empty cases.

Invalid model generations score as empty graphs; they are not removed from the benchmark. All main model configurations are evaluated against the same 1,024 filenames. Prediction repair and OOV handling must not change between compared configurations.

## What the metric measures

The score measures agreement with a compact, class-typed reference graph. It does not evaluate free-text attribute quality, instance localization, event boundaries, or speaker identity. Same-class instance swaps can receive full credit after alignment. A single reference can also omit valid alternative relations, so an unmatched prediction is not necessarily false in the media.

Audio-removal differences compare independently generated AV and video-only graphs under the evaluated protocol. They measure score sensitivity, not the number or fraction of relations that a model heard. Source-video bootstrap intervals describe source-sampling uncertainty conditional on the saved predictions and references; they do not cover decoding variability or annotation uncertainty.
