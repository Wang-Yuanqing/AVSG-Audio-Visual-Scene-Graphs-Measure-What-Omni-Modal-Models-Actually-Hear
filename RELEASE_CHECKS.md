# Release validation

The initial release was prepared against the current AVSG manuscript on
2026-10-29. Verification uses the exact paper-matched gold and saved predictions;
no fresh model inference or training is included in these checks.

| Check | Result |
|:--|:--|
| Unit tests: metric, parsing, malformed outputs and cohort coverage | 11 passed |
| Main-table F1 | 144/144 values match at three decimals |
| Main held-out table ALL/A2A entries | 22/22 match |
| Bootstrap metric estimates | 720 match the frozen scientific output exactly |
| Paired contrasts | 164 match exactly |
| Frame / seed comparisons | 18 match exactly |
| Frozen figure-data entries | 44/44 match regenerated estimates |
| Original prefill comparison | 1,023 clip alignments and 12 model subset results match |
| OOV-only bound | 0.6389338908 |
| Blinded gold edge audit | 106/114; design-weighted 0.9101877633 |
| Audit sampling | All 135 sampled edges reconstructed and verified |
| Training-data reconstruction | Both 827-record datasets match historical messages, targets, order and media basenames |
| Figure regeneration | Six generated vector figures match approved text and 96-dpi rendered pixels |

The benchmark reproduction was also run in an independent directory and a fresh
Python environment without third-party packages. It creates 24 main, 10 video-only,
5 baseline and 26 held-out rows, with the cohort written explicitly in each table.

Publication checks cover local documentation links, JSON/Python syntax, archive
provenance, file hashes, unwanted media/weight files, private participant fields,
machine-specific paths and known credential patterns. No original enrollment
records, private deployment notes, review correspondence, raw service wrappers,
or development Git history are included.

Inference entry points pass CPU-only command-line checks; the graph editor passes
a graph load/save roundtrip, and the audit demo builds successfully. Historical
model environments and the limits of recovered training/recovery sources are
documented in their respective READMEs. These checks do not establish a fresh GPU
rerun, model/API availability, or bitwise repeatability of stochastic generation.

Numerical evidence is recorded in `results/tables/`,
`results/analysis/verification.json`, and `training/reconstruction_check.json`.
`release_manifest.json` identifies every distributed file by SHA256.
