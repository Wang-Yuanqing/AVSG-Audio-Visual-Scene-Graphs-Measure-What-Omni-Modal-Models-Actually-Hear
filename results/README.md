# Saved predictions and scores

All prediction JSONL files are the exact bytes selected from the verified release
archive. Main-benchmark Nemotron instruct predictions use the paper-matched versions.
`source_manifest.json` records input provenance and SHA256 hashes; it does not contain
private deployment paths or participant enrollment information.

| Directory | Files | Clips per file | Evaluation |
|---|---:|---:|---|
| `predictions/main` | 24 | 1,024 | 12 configurations x raw/caption |
| `predictions/noaudio` | 10 | 1,024 | Raw video-only condition |
| `predictions/finetuned` | 2 | 197 | LoRA, AV input, test split |
| `predictions/finetuned_noaudio` | 2 | 197 | LoRA, video-only input, test split |
| `predictions/baselines` | 5 | 1,024 | Calibration predictions |

`raw` supplies video and audio without a reference caption. `caption` adds that
caption. A file in `noaudio/` retains its original `_raw` basename; the directory
identifies the video-only condition. Predictions retain OOV classes/relations and
documented failures. `healed` flags, where present, are historical recovery metadata.

```bash
python -m evaluation.reproduce
```

This recomputes `tables/main.csv`, `noaudio.csv`, `baselines.csv`, `heldout.csv`, and
`scores.json`. The JSON includes unrounded component counts and scores. The held-out
table evaluates all 12 raw untuned configurations, 10 video-only runs and four LoRA
conditions on the **same 197 clips**. Those scores should not be compared directly
with full-corpus scores as if they used the same cohort.

`expected_main_f1.json` contains the paper's 144 rounded main-table F1 values. The
reproduction command compares every value against this fixed reference and exits
with an error on any mismatch. Its tables report alignment fallback counts explicitly;
the original scorer's fallback is preserved rather than replacing archived results.

Point-estimate tables here are distinct from the source-video bootstrap analysis
under `analysis/` and `results/analysis/`.
